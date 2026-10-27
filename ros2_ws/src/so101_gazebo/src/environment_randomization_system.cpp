#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ignition/common/Console.hh>
#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/EventManager.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Light.hh>
#include <ignition/gazebo/components/LightCmd.hh>
#include <ignition/gazebo/components/Link.hh>
#include <ignition/gazebo/components/Model.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/gazebo/components/Sensor.hh>
#include <ignition/gazebo/components/Scene.hh>
#include <ignition/gazebo/components/Visual.hh>
#include <ignition/gazebo/components/VisualCmd.hh>
#include <ignition/gazebo/rendering/Events.hh>
#include <ignition/math/Color.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector2.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/msgs/boolean.pb.h>
#include <ignition/msgs/light.pb.h>
#include <ignition/msgs/stringmsg.pb.h>
#include <ignition/msgs/visual.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/rendering/Light.hh>
#include <ignition/rendering/RenderingIface.hh>
#include <ignition/rendering/Scene.hh>
#include <ignition/transport/Node.hh>

namespace so101::gazebo
{
using ignition::gazebo::Entity;
using ignition::gazebo::EntityComponentManager;
using ignition::gazebo::kNullEntity;
namespace components = ignition::gazebo::components;

struct Rgb
{
  double r{0.0};
  double g{0.0};
  double b{0.0};
};

struct LightCommand
{
  std::string name;
  double intensity{1.0};
  Rgb diffuse;
  ignition::math::Vector3d position;
  bool castShadows{false};
};

struct CameraDelta
{
  ignition::math::Vector3d position;
  ignition::math::Vector3d rpyDeg;
};

struct EpisodeCommand
{
  Rgb background;
  Rgb ground;
  Rgb table;
  Rgb target;
  Rgb goal;
  bool goalPositionEnabled{false};
  bool goalColorEnabled{false};
  bool goalSizeEnabled{false};
  ignition::math::Vector3d goalCenterWorld;
  ignition::math::Vector2d goalDeltaWorld;
  ignition::math::Vector2d goalHalfExtents;
  std::unordered_map<std::string, CameraDelta> cameras;
  std::vector<LightCommand> lights;
};

struct TargetCommand
{
  std::string modelName;
  Rgb color;
};

static std::vector<std::string> Split(
    const std::string &_value, const char _separator)
{
  std::vector<std::string> result;
  std::stringstream stream(_value);
  std::string item;
  while (std::getline(stream, item, _separator))
    result.push_back(item);
  return result;
}

static bool ParseDouble(const std::string &_text, double &_value)
{
  try
  {
    std::size_t consumed = 0;
    _value = std::stod(_text, &consumed);
    return consumed == _text.size() && std::isfinite(_value);
  }
  catch (...)
  {
    return false;
  }
}

static bool ParseRgb(const std::string &_text, Rgb &_rgb)
{
  const auto parts = Split(_text, ',');
  if (parts.size() != 3)
    return false;
  return ParseDouble(parts[0], _rgb.r) &&
         ParseDouble(parts[1], _rgb.g) &&
         ParseDouble(parts[2], _rgb.b);
}

static Entity ChildByName(
    EntityComponentManager &_ecm,
    const Entity _parent,
    const std::string &_name)
{
  const auto children = _ecm.ChildrenByComponents(
      _parent, components::Name(_name));
  if (children.size() != 1)
    return kNullEntity;
  return children.front();
}

static Entity ModelByName(
    EntityComponentManager &_ecm, const std::string &_name)
{
  return _ecm.EntityByComponents(
      components::Name(_name), components::Model());
}

static Entity LinkByName(
    EntityComponentManager &_ecm,
    const Entity _model,
    const std::string &_name)
{
  const auto entity = ChildByName(_ecm, _model, _name);
  if (entity == kNullEntity ||
      _ecm.Component<components::Link>(entity) == nullptr)
    return kNullEntity;
  return entity;
}

static Entity VisualByName(
    EntityComponentManager &_ecm,
    const Entity _link,
    const std::string &_name)
{
  const auto entity = ChildByName(_ecm, _link, _name);
  if (entity == kNullEntity ||
      _ecm.Component<components::Visual>(entity) == nullptr)
    return kNullEntity;
  return entity;
}

static void FillColor(
    ignition::msgs::Color *_message, const Rgb &_rgb)
{
  _message->set_r(_rgb.r);
  _message->set_g(_rgb.g);
  _message->set_b(_rgb.b);
  _message->set_a(1.0);
}

static bool MessageEqual(
    const ignition::msgs::Visual &_a,
    const ignition::msgs::Visual &_b)
{
  return _a.SerializeAsString() == _b.SerializeAsString();
}

static bool LightEqual(
    const ignition::msgs::Light &_a,
    const ignition::msgs::Light &_b)
{
  return _a.SerializeAsString() == _b.SerializeAsString();
}

class EnvironmentRandomizationSystem:
    public ignition::gazebo::System,
    public ignition::gazebo::ISystemConfigure,
    public ignition::gazebo::ISystemPreUpdate
{
  public: void Configure(
      const Entity &_entity,
      const std::shared_ptr<const sdf::Element> &,
      EntityComponentManager &_ecm,
      ignition::gazebo::EventManager &_eventManager) override
  {
    this->worldEntity = _entity;
    const auto *name = _ecm.Component<components::Name>(_entity);
    this->worldName = name == nullptr ? "empty" : name->Data();
    this->serviceName =
        "/world/" + this->worldName +
        "/so101_environment_randomization/apply";

    if (!this->node.Advertise(
            this->serviceName,
            &EnvironmentRandomizationSystem::OnRequest,
            this))
    {
      ignerr << "Failed to advertise SO-101 environment-randomization "
             << "service [" << this->serviceName << "]" << std::endl;
    }

    this->preRenderConnection =
        _eventManager.Connect<ignition::gazebo::events::PreRender>(
            std::bind(
                &EnvironmentRandomizationSystem::OnPreRender,
                this));

    ignmsg << "SO-101 environment-randomization service on ["
           << this->serviceName << "]" << std::endl;
  }

  private: bool OnRequest(
      const ignition::msgs::StringMsg &_request,
      ignition::msgs::Boolean &_response)
  {
    EpisodeCommand episode;
    bool hasEpisode = false;
    std::optional<TargetCommand> target;
    std::string error;

    std::stringstream stream(_request.data());
    std::string line;
    while (std::getline(stream, line))
    {
      if (line.empty())
        continue;
      const auto equal = line.find('=');
      if (equal == std::string::npos)
      {
        error = "missing '=' in line: " + line;
        break;
      }

      const auto key = line.substr(0, equal);
      const auto value = line.substr(equal + 1);

      if (key == "target_instance")
      {
        const auto parts = Split(value, ',');
        if (parts.size() != 4)
        {
          error = "target_instance expects model,r,g,b";
          break;
        }
        TargetCommand parsed;
        parsed.modelName = parts[0];
        if (!ParseDouble(parts[1], parsed.color.r) ||
            !ParseDouble(parts[2], parsed.color.g) ||
            !ParseDouble(parts[3], parsed.color.b))
        {
          error = "invalid target_instance color";
          break;
        }
        target = parsed;
        continue;
      }

      hasEpisode = true;

      if (key == "background")
      {
        if (!ParseRgb(value, episode.background))
          error = "invalid background RGB";
      }
      else if (key == "ground")
      {
        if (!ParseRgb(value, episode.ground))
          error = "invalid ground RGB";
      }
      else if (key == "table")
      {
        if (!ParseRgb(value, episode.table))
          error = "invalid table RGB";
      }
      else if (key == "target")
      {
        if (!ParseRgb(value, episode.target))
          error = "invalid target RGB";
      }
      else if (key == "goal_position_enabled")
      {
        if (value != "0" && value != "1")
          error = "goal_position_enabled must be 0 or 1";
        else
          episode.goalPositionEnabled = value == "1";
      }
      else if (key == "goal_color_enabled")
      {
        if (value != "0" && value != "1")
          error = "goal_color_enabled must be 0 or 1";
        else
          episode.goalColorEnabled = value == "1";
      }
      else if (key == "goal_size_enabled")
      {
        if (value != "0" && value != "1")
          error = "goal_size_enabled must be 0 or 1";
        else
          episode.goalSizeEnabled = value == "1";
      }
      else if (key == "goal_color")
      {
        if (!ParseRgb(value, episode.goal))
          error = "invalid goal RGB";
      }
      else if (key == "goal_center")
      {
        const auto parts = Split(value, ',');
        if (parts.size() != 3 ||
            !ParseDouble(parts[0], episode.goalCenterWorld.X()) ||
            !ParseDouble(parts[1], episode.goalCenterWorld.Y()) ||
            !ParseDouble(parts[2], episode.goalCenterWorld.Z()))
          error = "invalid goal_center";
      }
      else if (key == "goal_delta_xy")
      {
        const auto parts = Split(value, ',');
        if (parts.size() != 2 ||
            !ParseDouble(parts[0], episode.goalDeltaWorld.X()) ||
            !ParseDouble(parts[1], episode.goalDeltaWorld.Y()))
          error = "invalid goal_delta_xy";
      }
      else if (key == "goal_half_extents")
      {
        const auto parts = Split(value, ',');
        if (parts.size() != 2 ||
            !ParseDouble(parts[0], episode.goalHalfExtents.X()) ||
            !ParseDouble(parts[1], episode.goalHalfExtents.Y()) ||
            episode.goalHalfExtents.X() <= 0.0 ||
            episode.goalHalfExtents.Y() <= 0.0)
          error = "invalid goal_half_extents";
      }
      else if (key.rfind("camera.", 0) == 0)
      {
        const auto parts = Split(value, ',');
        if (parts.size() != 6)
        {
          error = "camera command expects 6 numeric values";
        }
        else
        {
          CameraDelta camera;
          double values[6]{};
          for (std::size_t i = 0; i < 6; ++i)
          {
            if (!ParseDouble(parts[i], values[i]))
            {
              error = "invalid camera command";
              break;
            }
          }
          if (error.empty())
          {
            camera.position.Set(values[0], values[1], values[2]);
            camera.rpyDeg.Set(values[3], values[4], values[5]);
            episode.cameras[key.substr(7)] = camera;
          }
        }
      }
      else if (key.rfind("light.", 0) == 0)
      {
        const auto parts = Split(value, ',');
        if (parts.size() != 8)
        {
          error = "light command expects 8 values";
        }
        else
        {
          LightCommand light;
          light.name = key.substr(6);
          double values[7]{};
          for (std::size_t i = 0; i < 7; ++i)
          {
            if (!ParseDouble(parts[i], values[i]))
            {
              error = "invalid light command";
              break;
            }
          }
          if (error.empty() && parts[7] != "0" && parts[7] != "1")
            error = "light shadow flag must be 0 or 1";
          if (error.empty())
          {
            light.intensity = values[0];
            light.diffuse = {values[1], values[2], values[3]};
            light.position.Set(values[4], values[5], values[6]);
            light.castShadows = parts[7] == "1";
            episode.lights.push_back(light);
          }
        }
      }
      else
      {
        error = "unknown randomization key: " + key;
      }

      if (!error.empty())
        break;
    }

    if (!error.empty())
    {
      ignerr << "Rejected SO-101 environment-randomization request: "
             << error << std::endl;
      _response.set_data(false);
      return true;
    }

    {
      std::lock_guard<std::mutex> lock(this->mutex);
      if (hasEpisode)
      {
        this->episodeCommand = episode;
        this->staticApplied = false;
        this->backgroundDirty = true;
        this->renderLightsDirty = true;
      }
      if (target.has_value())
        this->pendingTarget = target;
    }

    _response.set_data(true);
    return true;
  }

  public: void PreUpdate(
      const ignition::gazebo::UpdateInfo &,
      EntityComponentManager &_ecm) override
  {
    std::optional<EpisodeCommand> episode;
    std::optional<TargetCommand> target;
    bool needStatic = false;

    {
      std::lock_guard<std::mutex> lock(this->mutex);
      needStatic = this->episodeCommand.has_value() && !this->staticApplied;
      if (needStatic)
        episode = this->episodeCommand;
      if (this->pendingTarget.has_value())
        target = this->pendingTarget;
    }

    if (needStatic && episode.has_value())
    {
      if (this->ApplyStatic(*episode, _ecm))
      {
        std::lock_guard<std::mutex> lock(this->mutex);
        this->staticApplied = true;
      }
    }

    if (target.has_value())
    {
      if (this->ApplyTarget(*target, _ecm))
      {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (this->pendingTarget.has_value() &&
            this->pendingTarget->modelName == target->modelName)
        {
          this->pendingTarget.reset();
        }
      }
    }
  }

  private: bool ApplyStatic(
      const EpisodeCommand &_command,
      EntityComponentManager &_ecm)
  {
    bool ok = true;

    ok = this->ApplyVisualColor(
        _ecm, "ground_plane", "link", "visual",
        _command.ground) && ok;
    ok = this->ApplyVisualColor(
        _ecm, "lab_table", "table_link", "tabletop_visual",
        _command.table) && ok;

    const Entity goalModel = ModelByName(_ecm, "lab_table");
    const Entity goalLink = goalModel == kNullEntity
        ? kNullEntity
        : LinkByName(_ecm, goalModel, "goal_region_plate_link");
    const Entity goalVisual = goalLink == kNullEntity
        ? kNullEntity
        : VisualByName(_ecm, goalLink, "goal_region_plate_visual");

    if (goalVisual == kNullEntity)
    {
      ok = false;
    }
    else
    {
      auto *poseComponent = _ecm.Component<components::Pose>(goalVisual);
      if (poseComponent == nullptr)
      {
        ok = false;
      }
      else
      {
        if (!this->goalVisualBasePose.has_value())
          this->goalVisualBasePose = poseComponent->Data();

        if (_command.goalPositionEnabled)
        {
          auto pose = *this->goalVisualBasePose;
          const auto linkWorldPose = ignition::gazebo::worldPose(goalLink, _ecm);
          ignition::math::Vector3d worldDelta(
              _command.goalDeltaWorld.X(),
              _command.goalDeltaWorld.Y(),
              0.0);
          const auto localDelta =
              linkWorldPose.Rot().Inverse().RotateVector(worldDelta);
          pose.Pos() += localDelta;
          _ecm.SetComponentData<components::Pose>(goalVisual, pose);
        }
      }

      if (_command.goalColorEnabled || _command.goalSizeEnabled)
      {
        ignition::msgs::Visual visual;
        visual.set_id(goalVisual);
        if (_command.goalColorEnabled)
        {
          auto *material = visual.mutable_material();
          FillColor(material->mutable_ambient(), _command.goal);
          FillColor(material->mutable_diffuse(), _command.goal);
        }
        if (_command.goalSizeEnabled)
        {
          auto *geometry = visual.mutable_geometry();
          geometry->set_type(ignition::msgs::Geometry::BOX);
          auto *size = geometry->mutable_box()->mutable_size();
          size->set_x(2.0 * _command.goalHalfExtents.X());
          size->set_y(2.0 * _command.goalHalfExtents.Y());
          size->set_z(0.003);
        }
        this->SetVisualCommand(_ecm, goalVisual, visual);
      }
    }

    for (const auto &[sensorName, delta] : _command.cameras)
    {
      const Entity sensor = _ecm.EntityByComponents(
          components::Name(sensorName), components::Sensor());
      if (sensor == kNullEntity)
      {
        ok = false;
        continue;
      }

      auto *poseComponent = _ecm.Component<components::Pose>(sensor);
      if (poseComponent == nullptr)
      {
        ok = false;
        continue;
      }

      auto found = this->cameraBasePoses.find(sensorName);
      if (found == this->cameraBasePoses.end())
      {
        found = this->cameraBasePoses
            .emplace(sensorName, poseComponent->Data()).first;
      }

      auto pose = found->second;
      pose.Pos() += delta.position;
      constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
      const ignition::math::Quaterniond deltaRotation(
          delta.rpyDeg.X() * kDegToRad,
          delta.rpyDeg.Y() * kDegToRad,
          delta.rpyDeg.Z() * kDegToRad);
      pose.Rot() = pose.Rot() * deltaRotation;
      _ecm.SetComponentData<components::Pose>(sensor, pose);
    }

    for (const auto &light : _command.lights)
    {
      Entity entity = kNullEntity;
      _ecm.Each<components::Name, components::Light>(
          [&](const Entity &_entity,
              const components::Name *_name,
              const components::Light *) -> bool
          {
            if (_name != nullptr && _name->Data() == light.name)
            {
              entity = _entity;
              return false;
            }
            return true;
          });
      if (entity == kNullEntity)
      {
        ok = false;
        continue;
      }

      auto *poseComponent = _ecm.Component<components::Pose>(entity);
      if (poseComponent == nullptr)
      {
        ok = false;
        continue;
      }

      auto pose = poseComponent->Data();
      pose.Pos() = light.position;
      _ecm.SetComponentData<components::Pose>(entity, pose);

      ignition::msgs::Light message;
      message.set_id(entity);
      message.set_name(light.name);
      message.set_type(ignition::msgs::Light::POINT);
      message.set_cast_shadows(light.castShadows);
      auto *msgPose = message.mutable_pose();
      msgPose->mutable_position()->set_x(light.position.X());
      msgPose->mutable_position()->set_y(light.position.Y());
      msgPose->mutable_position()->set_z(light.position.Z());
      const Rgb scaledDiffuse{
          std::clamp(light.diffuse.r * light.intensity, 0.0, 1.0),
          std::clamp(light.diffuse.g * light.intensity, 0.0, 1.0),
          std::clamp(light.diffuse.b * light.intensity, 0.0, 1.0)};
      FillColor(message.mutable_diffuse(), scaledDiffuse);
      FillColor(message.mutable_specular(), {0.02, 0.02, 0.02});
      message.set_range(10.0);
      message.set_attenuation_constant(1.0);
      message.set_attenuation_linear(0.10);
      message.set_attenuation_quadratic(0.025);
      this->SetLightCommand(_ecm, entity, message);
    }

    const bool anyCastShadows = std::any_of(
        _command.lights.begin(), _command.lights.end(),
        [](const LightCommand &_light) { return _light.castShadows; });
    auto *sceneComponent = _ecm.Component<components::Scene>(this->worldEntity);
    if (sceneComponent != nullptr)
    {
      auto sceneData = sceneComponent->Data();
      sceneData.SetShadows(anyCastShadows);
      _ecm.SetComponentData<components::Scene>(this->worldEntity, sceneData);
    }

    return ok;
  }

  private: bool ApplyTarget(
      const TargetCommand &_command,
      EntityComponentManager &_ecm)
  {
    return this->ApplyVisualColor(
        _ecm,
        _command.modelName,
        "cube_link",
        "cube_visual",
        _command.color);
  }

  private: bool ApplyVisualColor(
      EntityComponentManager &_ecm,
      const std::string &_modelName,
      const std::string &_linkName,
      const std::string &_visualName,
      const Rgb &_color)
  {
    const Entity model = ModelByName(_ecm, _modelName);
    if (model == kNullEntity)
      return false;
    const Entity link = LinkByName(_ecm, model, _linkName);
    if (link == kNullEntity)
      return false;
    const Entity visualEntity = VisualByName(_ecm, link, _visualName);
    if (visualEntity == kNullEntity)
      return false;

    ignition::msgs::Visual visual;
    visual.set_id(visualEntity);
    auto *material = visual.mutable_material();
    FillColor(material->mutable_ambient(), _color);
    FillColor(material->mutable_diffuse(), _color);
    this->SetVisualCommand(_ecm, visualEntity, visual);
    return true;
  }

  private: void SetVisualCommand(
      EntityComponentManager &_ecm,
      const Entity _entity,
      const ignition::msgs::Visual &_message)
  {
    auto *component =
        _ecm.Component<components::VisualCmd>(_entity);
    if (component == nullptr)
    {
      _ecm.CreateComponent(
          _entity, components::VisualCmd(_message));
      return;
    }

    const auto state = component->SetData(
        _message, MessageEqual)
        ? ignition::gazebo::ComponentState::OneTimeChange
        : ignition::gazebo::ComponentState::NoChange;
    _ecm.SetChanged(
        _entity, components::VisualCmd::typeId, state);
  }

  private: void SetLightCommand(
      EntityComponentManager &_ecm,
      const Entity _entity,
      const ignition::msgs::Light &_message)
  {
    auto *component =
        _ecm.Component<components::LightCmd>(_entity);
    if (component == nullptr)
    {
      _ecm.CreateComponent(
          _entity, components::LightCmd(_message));
      return;
    }

    const auto state = component->SetData(
        _message, LightEqual)
        ? ignition::gazebo::ComponentState::OneTimeChange
        : ignition::gazebo::ComponentState::NoChange;
    _ecm.SetChanged(
        _entity, components::LightCmd::typeId, state);
  }

  private: void OnPreRender()
  {
    std::optional<Rgb> background;
    std::vector<LightCommand> lights;
    bool backgroundDirty = false;
    bool lightsDirty = false;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      if (!this->episodeCommand.has_value())
        return;
      backgroundDirty = this->backgroundDirty;
      lightsDirty = this->renderLightsDirty;
      if (backgroundDirty)
        background = this->episodeCommand->background;
      if (lightsDirty)
        lights = this->episodeCommand->lights;
    }

    auto scene = ignition::rendering::sceneFromFirstRenderEngine();
    if (!scene)
      return;

    if (backgroundDirty && background.has_value())
    {
      scene->SetBackgroundColor(
          ignition::math::Color(
              background->r, background->g, background->b, 1.0));
    }

    if (lightsDirty)
    {
      for (const auto &command : lights)
      {
        auto light = scene->LightByName(command.name);
        if (!light)
          continue;
        light->SetCastShadows(command.castShadows);
      }
    }

    std::lock_guard<std::mutex> lock(this->mutex);
    if (backgroundDirty)
      this->backgroundDirty = false;
    if (lightsDirty)
      this->renderLightsDirty = false;
  }

  private: Entity worldEntity{kNullEntity};
  private: std::string worldName{"empty"};
  private: std::string serviceName;
  private: ignition::transport::Node node;
  private: std::mutex mutex;
  private: std::optional<EpisodeCommand> episodeCommand;
  private: std::optional<TargetCommand> pendingTarget;
  private: bool staticApplied{false};
  private: bool backgroundDirty{false};
  private: bool renderLightsDirty{false};
  private: std::unordered_map<std::string, ignition::math::Pose3d>
      cameraBasePoses;
  private: std::optional<ignition::math::Pose3d> goalVisualBasePose;
  private: ignition::common::ConnectionPtr preRenderConnection;
};
}  // namespace so101::gazebo

IGNITION_ADD_PLUGIN(
    so101::gazebo::EnvironmentRandomizationSystem,
    ignition::gazebo::System,
    so101::gazebo::EnvironmentRandomizationSystem::ISystemConfigure,
    so101::gazebo::EnvironmentRandomizationSystem::ISystemPreUpdate)

IGNITION_ADD_PLUGIN_ALIAS(
    so101::gazebo::EnvironmentRandomizationSystem,
    "so101::gazebo::EnvironmentRandomizationSystem")
