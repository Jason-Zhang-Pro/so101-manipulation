#include <stdexcept>
#include <string>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/gazebo/components/World.hh>
#include "../src/grasp_attachment_core.hh"

namespace comp = ignition::gazebo::components;
using namespace so101::gazebo;
using ignition::gazebo::EntityComponentManager;
using ignition::gazebo::Entity;
using namespace std::chrono_literals;

class TestEntityComponentManager : public EntityComponentManager
{
  public: using EntityComponentManager::ProcessRemoveEntityRequests;
};

static void Check(bool _value, const char *_message)
{
  if (!_value) throw std::runtime_error(_message);
}

static Entity Model(EntityComponentManager &_ecm, Entity _world, const char *_name)
{
  const auto model = _ecm.CreateEntity();
  _ecm.CreateComponent(model, comp::Model());
  _ecm.CreateComponent(model, comp::Name(_name));
  _ecm.CreateComponent(model, comp::ParentEntity(_world));
  _ecm.SetParentEntity(model, _world);
  return model;
}

static Entity Link(EntityComponentManager &_ecm, Entity _model, const char *_name)
{
  const auto link = _ecm.CreateEntity();
  _ecm.CreateComponent(link, comp::Link());
  _ecm.CreateComponent(link, comp::Name(_name));
  _ecm.CreateComponent(link, comp::ParentEntity(_model));
  _ecm.SetParentEntity(link, _model);
  return link;
}

static std::string Command(const std::string &_id, const std::string &_action = "attach",
    const std::string &_target = "target_cube_123", int _attempt = 3)
{
  return "request_id=" + _id + "\naction=" + _action + "\ntarget=" + _target +
    "\nattempt=" + std::to_string(_attempt) + "\ncommand_rad=-0.01\nmeasured_rad=0.04";
}

int main()
{
  GraspAttachmentCommand parsed;
  std::string error;
  Check(ParseAttachmentCommand(Command("a"), parsed, error), "valid command rejected");
  Check(parsed.attempt == 3 && parsed.commandRad == -0.01, "numeric fields changed");
  for (const auto &bad : {Command("a") + "\naction=detach", Command("bad/id"),
       Command("a", "teleport"), Command("a", "attach", "other_model"),
       Command("a", "attach", "target_cube_123", -1), Command("a") + "\nunknown=x"})
    Check(!ParseAttachmentCommand(bad, parsed, error), "malformed command accepted");
  auto nan = Command("nan");
  nan.replace(nan.find("-0.01"), 5, "nan");
  Check(!ParseAttachmentCommand(nan, parsed, error), "NaN angle accepted");
  Check(AttachmentJsonString("a\"\n") == "\"a\\\"\\u000a\"", "JSON escape broken");

  TestEntityComponentManager ecm;
  const auto world = ecm.CreateEntity();
  ecm.CreateComponent(world, comp::World());
  const auto robot = Model(ecm, world, "so101");
  const auto fixed = Link(ecm, robot, "gripper_link");
  Link(ecm, robot, "moving_jaw_so101_v1_link");
  const auto cube = Model(ecm, world, "target_cube_123");
  const auto cubeLink = Link(ecm, cube, "cube_link");
  const ignition::math::Pose3d pose(0.2, 0.1, 0.8, 0.0, 0.1, 0.2);
  ecm.CreateComponent(cubeLink, comp::Pose(pose));
  GraspAttachmentCore core(world);
  ignition::gazebo::UpdateInfo info;
  info.paused = true;
  info.simTime = 2s;
  core.PreUpdate(info, ecm);
  core.PostUpdate(info, ecm);
  Check(core.Status().state == "detached", "dormant plugin altered world");
  Check(core.Submit(Command("first")).state == "queued", "attach not queued");
  Check(core.Submit(Command("first")).state == "queued", "queued retry not idempotent");
  Check(core.Submit(Command("conflict")).state == "error", "pending command overwritten");
  info.paused = false;
  core.PreUpdate(info, ecm);
  Check(core.Status().joint == ignition::gazebo::kNullEntity,
      "attachment mutated during unpaused physics");
  info.paused = true;
  core.PreUpdate(info, ecm);
  Check(core.Status().state == "queued", "attach acknowledged before PostUpdate");
  const auto joint = core.Status().joint;
  const auto *constraint = ecm.Component<comp::DetachableJoint>(joint);
  Check(constraint && constraint->Data().parentLink == fixed &&
      constraint->Data().childLink == cubeLink, "wrong fixed finger or target link");
  Check(ecm.Component<comp::Pose>(cubeLink)->Data() == pose, "attach teleported cube");
  core.PostUpdate(info, ecm);
  Check(core.Status().state == "attached" && *core.Status().attachTime == 2.0,
      "paused attachment confirmation failed");
  Check(core.Submit(Command("first")).state == "attached", "completed retry not idempotent");
  Check(core.Submit(Command("first", "detach")).state == "error", "id reuse accepted");
  Check(core.Submit(Command("wrong", "detach", "target_cube_456")).state == "queued",
      "detach must validate model in update thread");
  core.PreUpdate(info, ecm);
  Check(core.Status().state == "error" && ecm.HasEntity(joint), "wrong detach changed object");
  Check(core.Submit(Command("release", "detach")).state == "queued", "detach not queued");
  core.PreUpdate(info, ecm);
  core.PostUpdate(info, ecm);
  Check(core.Status().state == "queued", "detach ack preceded deferred removal");
  // Real server processes ECM removals between updates; this is an ECM unit
  // fixture only, not proof of DART constraint/contact behaviour.
  ecm.ProcessRemoveEntityRequests();
  core.PreUpdate(info, ecm);
  core.PostUpdate(info, ecm);
  Check(core.Status().state == "detached" && core.Status().joint == 0,
      "detached state not acknowledged after removal");
  Check(ecm.HasEntity(cube) && ecm.HasEntity(cubeLink), "detach removed the target");
  Check(core.Submit(Command("release", "detach")).state == "detached",
      "detach retry not idempotent");

  Check(core.Submit(Command("missing", "attach", "target_cube_missing")).state == "queued",
      "missing target request not queued");
  core.PreUpdate(info, ecm);
  Check(core.Status().state == "error", "missing target silently accepted");
  Check(core.Submit(Command("reattach")).state == "queued", "reattach rejected");
  core.PreUpdate(info, ecm);
  core.PostUpdate(info, ecm);
  Check(core.Status().state == "attached", "reattach failed");
  ecm.RequestRemoveEntity(core.Status().joint);
  ecm.ProcessRemoveEntityRequests();
  core.PostUpdate(info, ecm);
  Check(core.Status().state == "error", "lost constraint remained attached");
  Check(core.Status().Json().find("\"schema\":1") != std::string::npos,
      "response schema missing");
  return 0;
}
