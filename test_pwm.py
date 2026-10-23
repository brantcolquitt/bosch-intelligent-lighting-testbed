import Jetson.GPIO as GPIO
import time

# Use physical board pin numbering
PWM_PIN = 33 

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PWM_PIN, GPIO.OUT)
    
    # Initialize PWM on pin 33 at 100Hz
    pwm = GPIO.PWM(PWM_PIN, 100)
    pwm.start(0) # Start with 0% duty cycle
    
    print("Testing PWM on Pin 33. Press CTRL+C to stop.")
    try:
        while True:
            # Fade up
            for duty_cycle in range(0, 101, 5):
                pwm.ChangeDutyCycle(duty_cycle)
                time.sleep(0.05)
            # Fade down
            for duty_cycle in range(100, -1, -5):
                pwm.ChangeDutyCycle(duty_cycle)
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nTest stopped.")
    finally:
        pwm.stop()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
