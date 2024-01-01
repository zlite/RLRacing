This is a simplified Reinforcement Learning environment for a self-driving racecar. Right now it just lets you drive around the track with the arrow keys and shows the positing and heading, resetting if you go off the track. 

There are two version. One, which uses pygame, can be controlled by a human with a keyboard. The other, which is compabtible with the Gym/Gymnasium environment standard, does not have a human driving mode (because pygame is not supported)


* ToDo: add the DQN learning part
* ToDo: add randomly generated tracks and determine if we are on the track for any shape track
* ToDo: integrate with DonkeyCar so it uses GPS position rather than simulation coordinates
* ToDo: record a track in the real world with GPS positions
