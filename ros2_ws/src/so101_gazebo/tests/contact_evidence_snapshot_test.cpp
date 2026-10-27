#include <stdexcept>
#include <ignition/gazebo/components/Model.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include "../src/contact_evidence_snapshot.hh"

static void Check(bool _value, const char *_message)
{
  if (!_value)
    throw std::runtime_error(_message);
}

int main()
{
  namespace components = ignition::gazebo::components;
  using namespace std::chrono_literals;
  ignition::gazebo::EntityComponentManager ecm;
  const auto finger = ecm.CreateEntity();
  ecm.CreateComponent(finger, components::Collision());
  ecm.CreateComponent(finger, components::Name("finger_collision"));
  Check(!so101::gazebo::ContactEvidenceSnapshot(ecm, finger, 1500ms),
      "Missing physics data must remain unknown");
  ecm.CreateComponent(finger, components::ContactSensorData());
  const auto empty = so101::gazebo::ContactEvidenceSnapshot(ecm, finger, 1500ms);
  Check(empty && empty->contact_size() == 0, "Fresh empty data must be publishable");
  Check(empty->header().stamp().sec() == 1 && empty->header().stamp().nsec() == 500000000,
      "Empty data must carry physical source time");

  auto *data = ecm.Component<components::ContactSensorData>(finger);
  so101::gazebo::MarkContactPending(data->Data());
  Check(!so101::gazebo::ContactEvidenceSnapshot(ecm, finger, 1600ms),
      "Unrefreshed physics component must not fabricate a no-contact heartbeat");
  // Same update path as Fortress Physics::UpdateCollisions: even when the
  // equality function says NoChange, Component::SetData replaces the payload.
  data->SetData(ignition::msgs::Contacts{},
      [](const auto &, const auto &) { return true; });
  Check(so101::gazebo::ContactEvidenceSnapshot(ecm, finger, 1600ms).has_value(),
      "A fresh unchanged empty physical result must clear the pending marker");

  const auto cube = ecm.CreateEntity();
  ecm.CreateComponent(cube, components::Model());
  ecm.CreateComponent(cube, components::Name("target_cube_123"));
  const auto cubeCollision = ecm.CreateEntity();
  ecm.CreateComponent(cubeCollision, components::Collision());
  ecm.CreateComponent(cubeCollision, components::Name("collision"));
  ecm.CreateComponent(cubeCollision, components::ParentEntity(cube));
  ecm.SetParentEntity(cubeCollision, cube);
  auto *contact = data->Data().add_contact();
  contact->mutable_collision1()->set_id(finger);
  contact->mutable_collision2()->set_id(cubeCollision);
  const auto touching = so101::gazebo::ContactEvidenceSnapshot(ecm, finger, 1700ms);
  Check(touching && touching->contact_size() == 1, "Nonempty physical contact must survive");
  Check(touching->contact(0).collision2().name() == "target_cube_123::collision",
      "Entity IDs must resolve to target-scoped names without Physics names enabled");
  contact->mutable_collision2()->set_id(999999);
  Check(!so101::gazebo::ContactEvidenceSnapshot(ecm, finger, 1800ms),
      "Unresolvable collision identity must remain unknown");
  return 0;
}
