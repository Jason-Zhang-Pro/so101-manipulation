#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include <ignition/common/Console.hh>
#include <ignition/msgs/stringmsg.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>

#include "grasp_attachment_core.hh"

namespace so101::gazebo
{
class GraspAttachmentSystem final : public ignition::gazebo::System,
    public ignition::gazebo::ISystemConfigure,
    public ignition::gazebo::ISystemPreUpdate,
    public ignition::gazebo::ISystemPostUpdate
{
  public: void Configure(const ignition::gazebo::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &,
      ignition::gazebo::EntityComponentManager &_ecm,
      ignition::gazebo::EventManager &) override
  {
    const auto *name = _ecm.Component<attachment_components::Name>(_entity);
    if (!name)
    {
      ignerr << "GraspAttachmentSystem requires a named world.\n";
      return;
    }
    core = std::make_unique<GraspAttachmentCore>(_entity);
    const auto prefix = "/world/" + name->Data() + "/so101/grasp_attachment";
    if (!node.Advertise(prefix + "/command", &GraspAttachmentSystem::Command, this) ||
        !node.Advertise(prefix + "/status", &GraspAttachmentSystem::Status, this))
      ignerr << "Failed to advertise grasp attachment services at " << prefix << '\n';
    statePublisher = node.Advertise<ignition::msgs::StringMsg>(prefix + "/state");
    if (!statePublisher)
      ignerr << "Failed to advertise grasp attachment state at " << prefix << '\n';
  }

  public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
      ignition::gazebo::EntityComponentManager &_ecm) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (core) core->PreUpdate(_info, _ecm);
  }

  public: void PostUpdate(const ignition::gazebo::UpdateInfo &_info,
      const ignition::gazebo::EntityComponentManager &_ecm) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!core) return;
    core->PostUpdate(_info, _ecm);
    const auto &status = core->Status();
    const auto now = std::chrono::steady_clock::now();
    const bool stateChanged = status.command.requestId != lastRequestId ||
      status.state != lastState || status.joint != lastJoint || status.error != lastError;
    const bool pausedBoundary = _info.paused &&
      (!wasPaused || status.simTime != lastPublishedSimTime);
    const auto interval = _info.paused ? std::chrono::milliseconds(100) :
      std::chrono::milliseconds(33);
    // Publish changes and the exact paused step-end timestamp immediately.
    // While moving, throttle the 1 kHz physics callbacks to about 30 Hz wall
    // time. Paused 10 Hz heartbeats confirm commands without discovery churn.
    if (statePublisher && (stateChanged || pausedBoundary ||
        now - lastPublishedWallTime >= interval))
    {
      ignition::msgs::StringMsg message;
      message.set_data(status.Json());
      statePublisher.Publish(message);
      lastRequestId = status.command.requestId;
      lastState = status.state;
      lastError = status.error;
      lastJoint = status.joint;
      lastPublishedSimTime = status.simTime;
      lastPublishedWallTime = now;
    }
    wasPaused = _info.paused;
  }

  private: bool Command(const ignition::msgs::StringMsg &_request,
      ignition::msgs::StringMsg &_response)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!core) return false;
    _response.set_data(core->Submit(_request.data()).Json());
    return true;  // Application errors are explicit JSON, not transport failure.
  }

  private: bool Status(const ignition::msgs::StringMsg &,
      ignition::msgs::StringMsg &_response)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!core) return false;
    _response.set_data(core->Status().Json());
    return true;
  }

  private: std::mutex mutex;
  private: ignition::transport::Node node;
  private: ignition::transport::Node::Publisher statePublisher;
  private: std::string lastRequestId;
  private: std::string lastState;
  private: std::string lastError;
  private: AttachmentEntity lastJoint{ignition::gazebo::kNullEntity};
  private: double lastPublishedSimTime{-1.0};
  private: bool wasPaused{true};
  private: std::chrono::steady_clock::time_point lastPublishedWallTime{};
  private: std::unique_ptr<GraspAttachmentCore> core;
};
}  // namespace so101::gazebo

IGNITION_ADD_PLUGIN(so101::gazebo::GraspAttachmentSystem,
    ignition::gazebo::System,
    so101::gazebo::GraspAttachmentSystem::ISystemConfigure,
    so101::gazebo::GraspAttachmentSystem::ISystemPreUpdate,
    so101::gazebo::GraspAttachmentSystem::ISystemPostUpdate)
IGNITION_ADD_PLUGIN_ALIAS(so101::gazebo::GraspAttachmentSystem,
    "so101::gazebo::GraspAttachmentSystem")
