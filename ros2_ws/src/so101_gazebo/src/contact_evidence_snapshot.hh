#pragma once

#include <chrono>
#include <optional>
#include <string>

#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Collision.hh>
#include <ignition/gazebo/components/ContactSensorData.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/msgs/contacts.pb.h>

namespace so101::gazebo
{
// Fortress Physics::UpdateCollisions overwrites ContactSensorData even when
// empty. A PreUpdate marker lets us distinguish a fresh empty physics result
// from a component that exists but was never updated by a contact-capable engine.
inline constexpr const char *kPendingContactUpdate = "so101_pending_physics_update";

inline void MarkContactPending(ignition::msgs::Contacts &_data)
{
  _data.mutable_header()->Clear();
  _data.mutable_header()->add_data()->set_key(kPendingContactUpdate);
}

inline std::optional<ignition::msgs::Contacts> ContactEvidenceSnapshot(
    const ignition::gazebo::EntityComponentManager &_ecm,
    const ignition::gazebo::Entity _collision,
    const std::chrono::steady_clock::duration _time)
{
  namespace components = ignition::gazebo::components;
  if (!_ecm.Component<components::Collision>(_collision))
    return std::nullopt;
  const auto *data = _ecm.Component<components::ContactSensorData>(_collision);
  if (!data)
    return std::nullopt;
  for (const auto &entry : data->Data().header().data())
    if (entry.key() == kPendingContactUpdate)
      return std::nullopt;

  auto result = data->Data();
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(_time);
  auto *stamp = result.mutable_header()->mutable_stamp();
  stamp->set_sec(seconds.count());
  stamp->set_nsec(std::chrono::duration_cast<std::chrono::nanoseconds>(
      _time - seconds).count());
  for (auto &contact : *result.mutable_contact())
  {
    // Entity names are optional in Fortress's Physics configuration. Resolve
    // the physical IDs here so the ROS consumer can identify the current cube.
    for (auto *entity : {contact.mutable_collision1(), contact.mutable_collision2()})
    {
      if (!_ecm.Component<components::Collision>(entity->id()) ||
          !_ecm.Component<components::Name>(entity->id()))
        return std::nullopt;
      entity->set_name(ignition::gazebo::scopedName(entity->id(), _ecm, "::", false));
    }
    *contact.mutable_header()->mutable_stamp() = *stamp;
  }
  return result;
}
}  // namespace so101::gazebo
