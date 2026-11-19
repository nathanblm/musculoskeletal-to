#include <cctype>
#include <cmath>
#include <map>
#include <string>

#include <human_musculoskeletal/MuscleState.h>
#include <gazebo/gazebo_client.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>
#include <ros/ros.h>

class GazeboMuscleStateBridge
{
 public:
  GazeboMuscleStateBridge()
  {
    gazebo_node_.reset(new gazebo::transport::Node());
    gazebo_node_->Init();

    ros::NodeHandle private_node("~");
    std::string muscle_topic;
    private_node.param<std::string>("muscle_states_topic", muscle_topic,
        "/human/muscle_states");
    muscles_ = ros_node_.subscribe(muscle_topic, 1,
        &GazeboMuscleStateBridge::OnMuscles, this);
    ROS_INFO("Gazebo muscle fatigue plot topics: /human/muscle_fatigue/*");
  }

 private:
  static std::string TopicName(const std::string &name)
  {
    std::string result;
    for (const unsigned char ch : name)
    {
      if (std::isalnum(ch) || ch == '_')
        result += ch;
      else
        result += '_';
    }
    return result;
  }

  void Publish(const std::string &topic, double value)
  {
    if (!std::isfinite(value))
      return;
    auto &publisher = publishers_[topic];
    if (!publisher)
      publisher = gazebo_node_->Advertise<gazebo::msgs::Any>(topic);
    publisher->Publish(gazebo::msgs::ConvertAny(value));
  }

  void OnMuscles(const human_musculoskeletal::MuscleState::ConstPtr &msg)
  {
    if (msg->name.size() != msg->fatigue.size())
    {
      ROS_WARN_THROTTLE(5, "MuscleState names and fatigue differ in length");
      return;
    }
    for (std::size_t i = 0; i < msg->name.size(); ++i)
      Publish("/human/muscle_fatigue/" + TopicName(msg->name[i]),
          100.0 * msg->fatigue[i]);
  }

  ros::NodeHandle ros_node_;
  ros::Subscriber muscles_;
  gazebo::transport::NodePtr gazebo_node_;
  std::map<std::string, gazebo::transport::PublisherPtr> publishers_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gazebo_muscle_state_bridge");
  gazebo::client::setup();
  {
    GazeboMuscleStateBridge bridge;
    ros::spin();
  }
  gazebo::client::shutdown();
  return 0;
}
