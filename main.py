import serial
import time
import torch
import numpy as np
from PIL import Image
from enum import Enum
import logging
import os
import simpleaudio as sa
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fingerprint_system.log'),
        logging.StreamHandler()
    ]
)

class FingerPrintCommands(Enum):
    VERIFY_PASSWORD = b'\x13'
    GET_IMAGE = b'\x01'
    IMAGE_2_TZ = b'\x02'
    SEARCH = b'\x04'
    REG_MODEL = b'\x05'
    STORE = b'\x06'
    DELETE = b'\x0C'
    EMPTY = b'\x0D'

class DeepPrintModel(torch.nn.Module):
    def __init__(self):
        super(DeepPrintModel, self).__init__()
        # Feature extraction layers
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc = torch.nn.Linear(256 * 28 * 28, 512)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

class DeepPrintProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_deepprint_model()
        self.model = self.model.to(self.device)

    def load_deepprint_model(self):
        try:
            model = DeepPrintModel()
            if os.path.exists('deepprint_weights.pth'):
                model.load_state_dict(torch.load('deepprint_weights.pth'))
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Failed to load DeepPrint model: {e}")
            return None

    def extract_features(self, fingerprint_image):
        try:
            image = self.preprocess_image(fingerprint_image)
            with torch.no_grad():
                features = self.model(image)
            return features.cpu().numpy()
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return None

    def preprocess_image(self, image):
        image = image.resize((224, 224))
        image = np.array(image)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0).to(self.device)
        return image

    def compare_features(self, features1, features2):
        try:
            similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            return similarity
        except Exception as e:
            logging.error(f"Feature comparison failed: {e}")
            return 0.0

class SoundFeedback:
    def __init__(self):
        # Load sound files
        try:
            self.success_sound = sa.WaveObject.from_wave_file('success.wav')
            self.failure_sound = sa.WaveObject.from_wave_file('failure.wav')
        except Exception as e:
            logging.error(f"Failed to load sound files: {e}")
            self.success_sound = None
            self.failure_sound = None

    def play_success(self):
        if self.success_sound:
            self.success_sound.play()

    def play_failure(self):
        if self.failure_sound:
            self.failure_sound.play()

class R307FingerprintSensor:
    def __init__(self, port='/dev/ttyUSB0', baudrate=57600, password=0x00000000):
        try:
            self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=2.0)
            self.password = password
            self.deepprint = DeepPrintProcessor()
            self.feature_db = {}
            self.sound = SoundFeedback()
            logging.info("Fingerprint sensor initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize sensor: {e}")
            raise

    def calculate_checksum(self, packet):
        checksum = sum(packet) & 0xFFFF
        return checksum.to_bytes(2, 'big')

    def send_packet(self, command, data=b''):
        packet = (
            b'\xEF\x01'  # Header
            + self.password.to_bytes(4, 'big')  # Address
            + b'\x01'    # Package identifier
            + len(data + command).to_bytes(2, 'big')  # Length
            + command    # Command
            + data      # Data if any
        )
        packet += self.calculate_checksum(packet[6:])
        self.serial.write(packet)
        time.sleep(0.1)

    def read_response(self):
        if self.serial.in_waiting < 9:
            return None

        header = self.serial.read(9)
        if header[0:2] != b'\xEF\x01':
            return None

        length = int.from_bytes(header[7:9], 'big')
        data = self.serial.read(length - 2)
        checksum = self.serial.read(2)

        return {
            'confirmation_code': header[6],
            'data': data
        }

    def get_image_from_sensor(self):
        self.send_packet(FingerPrintCommands.GET_IMAGE.value)
        response = self.read_response()
        if response and response['confirmation_code'] == 0x00:
            # Convert sensor data to image
            # Implementation depends on sensor's image format
            return Image.fromarray(response['data'])
        return None

    def enroll_finger(self, finger_id):
        logging.info(f"Starting enrollment for ID {finger_id}")
        print("Place finger on sensor...")
        
        # First image
        self.send_packet(FingerPrintCommands.GET_IMAGE.value)
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to capture first image"

        # Convert first image
        self.send_packet(FingerPrintCommands.IMAGE_2_TZ.value, b'\x01')
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to process first image"

        print("Remove finger...")
        time.sleep(2)
        print("Place same finger again...")

        # Second image
        self.send_packet(FingerPrintCommands.GET_IMAGE.value)
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to capture second image"

        # Convert second image
        self.send_packet(FingerPrintCommands.IMAGE_2_TZ.value, b'\x02')
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to process second image"

        # Create model
        self.send_packet(FingerPrintCommands.REG_MODEL.value)
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to create model"

        # Get image for DeepPrint
        image = self.get_image_from_sensor()
        if image:
            features = self.deepprint.extract_features(image)
            if features is not None:
                self.feature_db[finger_id] = features
                logging.info(f"DeepPrint features extracted for ID {finger_id}")

        # Store model
        data = bytes([finger_id >> 8, finger_id & 0xFF, 0x00, 0x00])
        self.send_packet(FingerPrintCommands.STORE.value, data)
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to store template"

        self.sound.play_success()
        logging.info(f"Enrollment successful for ID {finger_id}")
        return True, "Enrollment successful"

    def verify_finger(self):
        logging.info("Starting fingerprint verification")
        print("Place finger to verify...")
        
        # Get image
        self.send_packet(FingerPrintCommands.GET_IMAGE.value)
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to capture image"

        # Convert image
        self.send_packet(FingerPrintCommands.IMAGE_2_TZ.value, b'\x01')
        response = self.read_response()
        if not response or response['confirmation_code'] != 0x00:
            self.sound.play_failure()
            return False, "Failed to process image"

        # Verify with sensor
        self.send_packet(FingerPrintCommands.SEARCH.value, b'\x01\x00\x00\x00\x05')
        response = self.read_response()
        
        # Also verify with DeepPrint if available
        image = self.get_image_from_sensor()
        if image and self.feature_db:
            features = self.deepprint.extract_features(image)
            if features is not None:
                best_match = 0
                best_id = None
                for fid, stored_features in self.feature_db.items():
                    similarity = self.deepprint.compare_features(features, stored_features)
                    if similarity > best_match:
                        best_match = similarity
                        best_id = fid
                
                if best_match > 0.8:
                    logging.info(f"DeepPrint verification successful (ID: {best_id})")

        if response and response['confirmation_code'] == 0x00:
            self.sound.play_success()
            logging.info("Verification successful")
            return True, "Fingerprint verified"
        else:
            self.sound.play_failure()
            logging.info("Verification failed")
            return False, "Verification failed"

    def delete_finger(self, finger_id):
        logging.info(f"Deleting fingerprint ID {finger_id}")
        data = bytes([finger_id >> 8, finger_id & 0xFF, 0x00, 0x01])
        self.send_packet(FingerPrintCommands.DELETE.value, data)
        response = self.read_response()
        success = response and response['confirmation_code'] == 0x00
        
        if success:
            if finger_id in self.feature_db:
                del self.feature_db[finger_id]
            self.sound.play_success()
            logging.info(f"Successfully deleted ID {finger_id}")
        else:
            self.sound.play_failure()
            logging.info(f"Failed to delete ID {finger_id}")
            
        return success

    def empty_database(self):
        logging.info("Emptying fingerprint database")
        self.send_packet(FingerPrintCommands.EMPTY.value)
        response = self.read_response()
        success = response and response['confirmation_code'] == 0x00
        
        if success:
            self.feature_db.clear()
            self.sound.play_success()
            logging.info("Database emptied successfully")
        else:
            self.sound.play_failure()
            logging.info("Failed to empty database")
            
        return success

    def save_features_database(self, filename='fingerprint_features.npy'):
        try:
            np.save(filename, self.feature_db)
            logging.info(f"Features database saved to {filename}")
            return True
        except Exception as e:
            logging.error(f"Failed to save features database: {e}")
            return False

    def load_features_database(self, filename='fingerprint_features.npy'):
        try:
            if os.path.exists(filename):
                self.feature_db = np.load(filename, allow_pickle=True).item()
                logging.info(f"Features database loaded from {filename}")
                return True
        except Exception as e:
            logging.error(f"Failed to load features database: {e}")
        return False

def main():
    try:
        # Initialize sensor with appropriate port
        sensor = R307FingerprintSensor(port='/dev/ttyUSB0')  # Adjust port as needed
        
        # Load existing features database if available
        sensor.load_features_database()

        while True:
            print("\nFingerprint Sensor Menu:")
            print("1. Enroll new fingerprint")
            print("2. Verify fingerprint")
            print("3. Delete fingerprint")
            print("4. Empty database")
            print("5. Save features database")
            print("6. Exit")

            choice = input("Select an option: ")

            if choice == '1':
                try:
                    finger_id = int(input("Enter ID for new fingerprint (1-999): "))
                    success, message = sensor.enroll_finger(finger_id)
                    print(message)
                except ValueError:
                    print("Please enter a valid number")

            elif choice == '2':
                success, message = sensor.verify_finger()
                print(message)

            elif choice == '3':
                try:
                    finger_id = int(input("Enter ID to delete: "))
                    if sensor.delete_finger(finger_id):
                        print("Fingerprint deleted")
                    else:
                        print("Failed to delete fingerprint")
                except ValueError:
                    print("Please enter a valid number")

            elif choice == '4':
                if sensor.empty_database():
                    print("Database emptied")
                else:
                    print("Failed to empty database")

            elif choice == '5':
                if sensor.save_features_database():
                    print("Features database saved")
                else:
                    print("Failed to save features database")

            elif choice == '6':
                print("Exiting...")
                break

            else:
                print("Invalid option")

    except Exception as e:
        logging.error(f"Program error: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()