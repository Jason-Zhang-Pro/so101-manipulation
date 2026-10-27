#ifndef SO101_GAZEBO_GRASP_ATTACHMENT_CORE_HH_
#define SO101_GAZEBO_GRASP_ATTACHMENT_CORE_HH_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/components/DetachableJoint.hh>
#include <ignition/gazebo/components/Link.hh>
#include <ignition/gazebo/components/Model.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include <ignition/gazebo/components/Static.hh>

namespace so101::gazebo
{
namespace attachment_components = ignition::gazebo::components;
using AttachmentEntity = ignition::gazebo::Entity;

struct GraspAttachmentCommand
{
  std::string requestId;
  std::string action;
  std::string target;
  std::int64_t attempt{-1};
  double commandRad{0};
  double measuredRad{0};

  bool operator==(const GraspAttachmentCommand &_other) const
  {
    return requestId == _other.requestId && action == _other.action &&
      target == _other.target && attempt == _other.attempt &&
      commandRad == _other.commandRad && measuredRad == _other.measuredRad;
  }
};

inline bool AttachmentSafeName(const std::string &_name)
{
  return !_name.empty() && _name.size() <= 128 &&
    _name.find_first_not_of(
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") ==
      std::string::npos;
}

inline bool ParseAttachmentCommand(const std::string &_payload,
    GraspAttachmentCommand &_command, std::string &_error)
{
  if (_payload.size() > 2048)
  {
    _error = "command exceeds 2048 bytes";
    return false;
  }
  std::map<std::string, std::string> fields;
  std::istringstream stream(_payload);
  std::string line;
  while (std::getline(stream, line))
  {
    const auto delimiter = line.find('=');
    if (delimiter == std::string::npos ||
        !fields.emplace(line.substr(0, delimiter), line.substr(delimiter + 1)).second)
    {
      _error = "expected unique key=value lines";
      return false;
    }
  }
  for (const auto *key : {"request_id", "action", "target", "attempt",
                          "command_rad", "measured_rad"})
  {
    if (fields.count(key) != 1)
    {
      _error = std::string("missing field: ") + key;
      return false;
    }
  }
  if (fields.size() != 6)
  {
    _error = "unknown command field";
    return false;
  }
  _command.requestId = fields.at("request_id");
  _command.action = fields.at("action");
  _command.target = fields.at("target");
  if (!AttachmentSafeName(_command.requestId) ||
      !AttachmentSafeName(_command.target) ||
      _command.target.rfind("target_cube_", 0) != 0 ||
      (_command.action != "attach" && _command.action != "detach"))
  {
    _error = "invalid request_id, action, or target_cube_ model name";
    return false;
  }
  try
  {
    std::size_t used = 0;
    const auto &attempt = fields.at("attempt");
    if (attempt.empty() || attempt.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("attempt");
    _command.attempt = std::stoll(attempt, &used);
    if (used != attempt.size() || _command.attempt < 0)
      throw std::invalid_argument("attempt");
    const auto parseAngle = [](const std::string &_text)
    {
      std::size_t consumed = 0;
      const double value = std::stod(_text, &consumed);
      if (consumed != _text.size() || !std::isfinite(value))
        throw std::invalid_argument("angle");
      return value;
    };
    _command.commandRad = parseAngle(fields.at("command_rad"));
    _command.measuredRad = parseAngle(fields.at("measured_rad"));
  }
  catch (const std::exception &)
  {
    _error = "attempt must be a nonnegative integer and angles must be finite";
    return false;
  }
  _error.clear();
  return true;
}

inline std::string AttachmentJsonString(const std::string &_text)
{
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : _text)
  {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 0x20)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    else
      out << c;
  }
  out << '"';
  return out.str();
}

struct GraspAttachmentStatus
{
  GraspAttachmentCommand command;
  std::string state{"detached"};
  AttachmentEntity joint{ignition::gazebo::kNullEntity};
  double simTime{0};
  std::optional<double> attachTime;
  std::string error;

  std::string Json() const
  {
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"schema\":1,\"request_id\":" << AttachmentJsonString(command.requestId)
        << ",\"state\":" << AttachmentJsonString(state)
        << ",\"target\":" << AttachmentJsonString(command.target)
        << ",\"attempt\":" << command.attempt
        << ",\"joint_entity\":" << joint
        << ",\"sim_time_s\":" << simTime << ",\"attach_sim_time_s\":";
    if (attachTime) out << *attachTime; else out << "null";
    out << ",\"command_rad\":" << command.commandRad
        << ",\"measured_rad\":" << command.measuredRad
        << ",\"error\":" << AttachmentJsonString(error) << '}';
    return out.str();
  }
};

// All methods are called under the plugin's mutex. Only PreUpdate mutates ECM.
// "attached" acknowledges a DetachableJoint applied through Physics::Update;
// it is not an independent measurement of grasp contact or constraint accuracy.
class GraspAttachmentCore
{
  public: explicit GraspAttachmentCore(AttachmentEntity _world) : world(_world) {}

  public: GraspAttachmentStatus Submit(const std::string &_payload)
  {
    GraspAttachmentCommand command;
    std::string error;
    if (!ParseAttachmentCommand(_payload, command, error))
      return Rejection(command, error);
    const auto previous = history.find(command.requestId);
    if (previous != history.end())
      return previous->second.first == command ? previous->second.second :
        Rejection(command, "request_id was already used for different contents");
    if (pending)
      return *pending == command ? status :
        Rejection(command, "another attachment command is pending");
    pending = command;
    status.command = command;
    status.state = "queued";
    status.joint = joint;
    status.error.clear();
    if (command.action == "attach" && joint == ignition::gazebo::kNullEntity)
      status.attachTime.reset();
    phase = Phase::Queued;
    return status;
  }

  public: const GraspAttachmentStatus &Status() const { return status; }

  public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
      ignition::gazebo::EntityComponentManager &_ecm)
  {
    status.simTime = std::chrono::duration<double>(_info.simTime).count();
    // Paused updates still execute this method. Never alter attachment while a
    // client is allowing physics to advance; detached ACK precedes next step.
    if (!pending || phase != Phase::Queued || !_info.paused)
      return;
    const auto &command = *pending;
    if (command.action == "detach")
    {
      if (joint != ignition::gazebo::kNullEntity &&
          (command.target != active.target || command.attempt != active.attempt))
      {
        Fail("detach target/attempt does not match the attached object");
        return;
      }
      if (joint != ignition::gazebo::kNullEntity)
        _ecm.RequestRemoveEntity(joint);
      phase = Phase::DetachRemoval;
      return;
    }
    if (joint != ignition::gazebo::kNullEntity)
    {
      if (command.target != active.target || command.attempt != active.attempt)
      {
        Fail("a different target/attempt is already attached");
        return;
      }
      // Repeated attach cannot overwrite the angle captured by the first grasp.
      status.command.commandRad = active.commandRad;
      status.command.measuredRad = active.measuredRad;
      phase = Phase::AttachPost;
      return;
    }
    const auto parentModel = Model(_ecm, "so101");
    const auto childModel = Model(_ecm, command.target);
    parent = Link(_ecm, parentModel, "gripper_link");
    child = Link(_ecm, childModel, "cube_link");
    if (parent == ignition::gazebo::kNullEntity || child == ignition::gazebo::kNullEntity)
    {
      Fail("so101/gripper_link or requested target/cube_link is missing or ambiguous");
      return;
    }
    const auto *isStatic = _ecm.Component<attachment_components::Static>(childModel);
    if (isStatic && isStatic->Data())
    {
      Fail("target model must be dynamic");
      return;
    }
    joint = _ecm.CreateEntity();
    _ecm.CreateComponent(joint, attachment_components::DetachableJoint(
      {parent, child, "fixed"}));
    // No PoseCmd, joint reset, collision change, or model removal is performed.
    active = command;
    status.joint = joint;
    status.attachTime = status.simTime;
    phase = Phase::AttachPost;
  }

  public: void PostUpdate(const ignition::gazebo::UpdateInfo &_info,
      const ignition::gazebo::EntityComponentManager &_ecm)
  {
    status.simTime = std::chrono::duration<double>(_info.simTime).count();
    if (pending && phase == Phase::AttachPost)
    {
      if (!_ecm.Component<attachment_components::DetachableJoint>(joint) ||
          !_ecm.HasEntity(parent) || !_ecm.HasEntity(child))
        Fail("attachment entities disappeared before confirmation");
      else
        Complete("attached");
    }
    else if (pending && phase == Phase::DetachRemoval)
    {
      // RequestRemoveEntity is deferred. Only acknowledge after ECM completed
      // removal, so Physics has had an update to detach the actual DART joint.
      if (joint == ignition::gazebo::kNullEntity || !_ecm.HasEntity(joint))
      {
        joint = parent = child = ignition::gazebo::kNullEntity;
        status.joint = joint;
        Complete("detached");
      }
    }
    else if (!pending && joint != ignition::gazebo::kNullEntity &&
        (!_ecm.HasEntity(joint) || !_ecm.HasEntity(parent) || !_ecm.HasEntity(child)))
    {
      status.state = "error";
      status.error = "attached entity disappeared unexpectedly";
    }
  }

  private: enum class Phase { Idle, Queued, AttachPost, DetachRemoval };

  private: static AttachmentEntity Model(ignition::gazebo::EntityComponentManager &_ecm,
      const std::string &_name, AttachmentEntity _world)
  {
    const auto matches = _ecm.ChildrenByComponents(_world,
      attachment_components::Model(), attachment_components::Name(_name));
    return matches.size() == 1 ? matches.front() : ignition::gazebo::kNullEntity;
  }
  private: AttachmentEntity Model(ignition::gazebo::EntityComponentManager &_ecm,
      const std::string &_name) const { return Model(_ecm, _name, world); }

  private: static AttachmentEntity Link(ignition::gazebo::EntityComponentManager &_ecm,
      AttachmentEntity _model, const std::string &_name)
  {
    if (_model == ignition::gazebo::kNullEntity)
      return ignition::gazebo::kNullEntity;
    const auto matches = _ecm.ChildrenByComponents(_model,
      attachment_components::Link(), attachment_components::Name(_name));
    return matches.size() == 1 ? matches.front() : ignition::gazebo::kNullEntity;
  }

  private: GraspAttachmentStatus Rejection(const GraspAttachmentCommand &_command,
      const std::string &_error) const
  {
    auto rejected = status;
    rejected.command = _command;
    rejected.state = "error";
    rejected.error = _error;
    return rejected;
  }

  private: void Fail(const std::string &_error)
  {
    status.error = _error;
    Complete("error");
  }

  private: void Complete(const std::string &_state)
  {
    status.state = _state;
    history[status.command.requestId] = {*pending, status};
    completed.push_back(status.command.requestId);
    if (completed.size() > 128)
    {
      history.erase(completed.front());
      completed.pop_front();
    }
    pending.reset();
    phase = Phase::Idle;
  }

  private: AttachmentEntity world;
  private: AttachmentEntity joint{ignition::gazebo::kNullEntity};
  private: AttachmentEntity parent{ignition::gazebo::kNullEntity};
  private: AttachmentEntity child{ignition::gazebo::kNullEntity};
  private: GraspAttachmentCommand active;
  private: GraspAttachmentStatus status;
  private: std::optional<GraspAttachmentCommand> pending;
  private: Phase phase{Phase::Idle};
  private: std::map<std::string,
    std::pair<GraspAttachmentCommand, GraspAttachmentStatus>> history;
  private: std::deque<std::string> completed;
};
}  // namespace so101::gazebo

#endif
