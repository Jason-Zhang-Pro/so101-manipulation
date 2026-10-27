#include <array>
#include <chrono>
#include <memory>
#include <string>

#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/components/ContactSensor.hh>
#include <ignition/gazebo/components/Link.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>

#include "contact_evidence_snapshot.hh"

namespace so101::gazebo
{
namespace components = ignition::gazebo::components;
using ignition::gazebo::Entity;
using ignition::gazebo::EntityComponentManager;
using ignition::gazebo::kNullEntity;

class ContactEvidenceSystem final : public ignition::gazebo::System,
    public ignition::gazebo::ISystemConfigure,
    public ignition::gazebo::ISystemPreUpdate,
    public ignition::gazebo::ISystemPostUpdate
{
  struct Channel
  {
    std::string sensorName;
    std::string topic;
    Entity collision{kNullEntity};
    ignition::transport::Node::Publisher publisher;
  };

  public: void Configure(const Entity &, const std::shared_ptr<const sdf::Element> &,
      EntityComponentManager &, ignition::gazebo::EventManager &) override
  {
    for (auto &channel : this->channels)
      channel.publisher = this->node.Advertise<ignition::msgs::Contacts>(channel.topic);
  }

  public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
      EntityComponentManager &_ecm) override
  {
    if (_info.paused || _info.dt <= std::chrono::steady_clock::duration::zero())
      return;
    for (auto &channel : this->channels)
    {
      if (!_ecm.Component<components::Collision>(channel.collision))
        channel.collision = this->ResolveCollision(_ecm, channel.sensorName);
      // The native Contact system creates this component. Do not create a
      // fabricated empty stream if its sensor collision failed to resolve.
      if (auto *data = _ecm.Component<components::ContactSensorData>(channel.collision))
        MarkContactPending(data->Data());
    }
  }

  public: void PostUpdate(const ignition::gazebo::UpdateInfo &_info,
      const EntityComponentManager &_ecm) override
  {
    if (_info.paused || _info.dt <= std::chrono::steady_clock::duration::zero())
      return;
    for (auto &channel : this->channels)
    {
      const auto sample = ContactEvidenceSnapshot(_ecm, channel.collision, _info.simTime);
      if (sample)
        channel.publisher.Publish(*sample);  // Deliberately includes zero contacts.
    }
  }

  private: static Entity ResolveCollision(EntityComponentManager &_ecm,
      const std::string &_sensorName)
  {
    Entity result = kNullEntity;
    unsigned matches = 0;
    _ecm.Each<components::ContactSensor, components::Name, components::ParentEntity>(
        [&](const Entity &, const components::ContactSensor *_sensor,
            const components::Name *_name, const components::ParentEntity *_parent)
        {
          if (_name->Data() != _sensorName)
            return true;
          ++matches;
          const auto sensor = _sensor->Data();
          if (!_ecm.Component<components::Link>(_parent->Data()) ||
              !sensor || !sensor->HasElement("contact"))
            return true;
          const auto contact = sensor->GetElement("contact");
          if (!contact->HasElement("collision"))
            return true;
          const auto collision = contact->GetElement("collision");
          if (collision->GetNextElement("collision"))
            return true;
          const auto children = _ecm.ChildrenByComponents(_parent->Data(),
              components::Collision(), components::Name(collision->Get<std::string>()));
          if (children.size() == 1)
            result = children.front();
          return true;
        });
    return matches == 1 ? result : kNullEntity;
  }

  private: ignition::transport::Node node;
  private: std::array<Channel, 2> channels{{
    {"fixed_finger_contact", "/so101/contacts/fixed_finger"},
    {"moving_finger_contact", "/so101/contacts/moving_finger"}
  }};
};
}  // namespace so101::gazebo

IGNITION_ADD_PLUGIN(so101::gazebo::ContactEvidenceSystem,
    ignition::gazebo::System,
    so101::gazebo::ContactEvidenceSystem::ISystemConfigure,
    so101::gazebo::ContactEvidenceSystem::ISystemPreUpdate,
    so101::gazebo::ContactEvidenceSystem::ISystemPostUpdate)
IGNITION_ADD_PLUGIN_ALIAS(so101::gazebo::ContactEvidenceSystem,
    "so101::gazebo::ContactEvidenceSystem")
