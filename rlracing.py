import numpy as np
import pygame

class RaceTrackEnv:
    def __init__(self):
        pygame.init()
        self.track_width = 20
        self.field_size = (1000, 1000)
        self.car_size = (10, 15)  # Set the size of the car here
        self.oval_center = (500, 500)
        self.oval_axes = (300, 400)
        self.car_speed = 5
        self.car_angle = 0
        self.turn_angle = 5
        self.car_image_path = 'car.png'

        # Load the car image, scale it to the car size, and get its size
        self.car_image_original = pygame.image.load(self.car_image_path)
        self.car_image_original = pygame.transform.scale(self.car_image_original, self.car_size)  # Scale image to the car size
        self.car_image = self.car_image_original.copy()

        # Calculate starting position on the track (centered on the top of the oval)
        self.start_position = [self.oval_center[0], -15 + self.oval_center[1] - self.oval_axes[1] + self.track_width/2 + self.car_size[1]/2]
        self.car_position = self.start_position.copy()

    def reset(self):
        self.car_position = self.start_position.copy()
        self.car_angle = 0
        self.car_image = self.car_image_original.copy()
        return self.get_state()
    
    def step(self, action):
        if action == 0:  # Turn left
            self.car_angle -= self.turn_angle
        elif action == 1:  # Turn right
            self.car_angle += self.turn_angle

        dx = self.car_speed * np.cos(np.radians(self.car_angle))
        dy = self.car_speed * np.sin(np.radians(self.car_angle))
        self.car_position[0] += dx
        self.car_position[1] += dy

        done = not self.is_on_track()
        reward = 1 if not done else -100

        return self.get_state(), reward, done

    def is_on_track(self):
        x, y = self.car_position
        cx, cy = self.oval_center
        a, b = self.oval_axes

        outer_ellipse = ((x - cx) / (a + self.track_width/2))**2 + ((y - cy) / (b + self.track_width/2))**2
        inner_ellipse = ((x - cx) / (a - self.track_width/2))**2 + ((y - cy) / (b - self.track_width/2))**2
        return outer_ellipse <= 1 and inner_ellipse >= 1

    def get_state(self):
        return np.array(self.car_position + [self.car_angle])

    def rotate_car(self):
        """Rotate the car surface around its center and adjust the car's angle."""
        rotated_image = pygame.transform.rotate(self.car_image_original, -self.car_angle + 90)  # Correct the orientation by addingh 90 degrees
        rotated_rect = rotated_image.get_rect(center=self.car_position)
        return rotated_image, rotated_rect

    def render(self, screen):
        screen.fill((255, 255, 255))  # White background

        # Draw the track
        pygame.draw.ellipse(screen, (0, 0, 255), [
            self.oval_center[0] - self.oval_axes[0] - self.track_width/2,
            self.oval_center[1] - self.oval_axes[1] - self.track_width/2,
            2 * (self.oval_axes[0] + self.track_width/2),
            2 * (self.oval_axes[1] + self.track_width/2)
        ], self.track_width)

        # Rotate and draw the car
        rotated_image, rotated_rect = self.rotate_car()
        screen.blit(rotated_image, rotated_rect.topleft)

        pygame.display.flip()  # Update the full display Surface to the screen

    def run(self):
        screen = pygame.display.set_mode(self.field_size)
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        state, reward, done = self.step(0)
                    elif event.key == pygame.K_RIGHT:
                        state, reward, done = self.step(1)
                    else:
                        continue  # Ignore other keys

                    print(f"State: {state}, Reward: {reward}, Done: {done}")
                    if done:
                        print("Car is off the track! Resetting environment.")
                        self.reset()

            self.render(screen)
            clock.tick(30)  # Run at 30 frames per second

        pygame.quit()

# Example usage
env = RaceTrackEnv()
env.reset()
env.run()
