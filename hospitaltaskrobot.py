import time
import pyttsx3
from random import randint
from transformers import pipeline

def verbal_instruction_to_action(instruction):
    """
    Parses the verbal instruction into an actionable plan using a Hugging Face transformer.
    """
    # Initialize Hugging Face pipeline for text generation
    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    # Engineer a prompt to extract task, object, destination, and time
    prompt = (
        f"I am a hospital robot receiving this instruction from a staff member. \n"
        f"Extract the following information from the instruction: \n"
        f"Instruction: '{instruction}' \n"
        f"Provide: \n"
        f"1. Task \n"
        f"2. Object \n"
        f"3. Destination \n"
        f"4. Time"
    )

    response = generator(prompt, max_length=50)
    extracted_info = response[0]['generated_text']

    # Parse the generated text into a dictionary
    try:
        info_lines = extracted_info.split("\n")
        parsed_info = {}
        for line in info_lines:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_info[key.strip().lower()] = value.strip()

        # Return the actionable plan
        return {
            "task": parsed_info.get("task", "unknown"),
            "object": parsed_info.get("object", "unknown"),
            "destination": parsed_info.get("destination", "unknown"),
            "time": parsed_info.get("time", "unknown"),
            "bin_location": f"closet_{parsed_info.get('object', 'unknown').replace(' ', '_')}_bin"
        }
    except Exception as e:
        print("Error parsing response from transformer:", e)
        return None

def move_to_location(location):
    """
    Plan a path to the given location using NVIDIA Isaac Sim and execute it.
    """
    # Simulating a map of the hospital
    hospital_map = {
        "closet_bandage_bin": (5, 10),
        "nurse_station": (0, 0),
        "room_205": (15, 20)
    }

    if location not in hospital_map:
        print(f"Error: Unknown location {location}.")
        return "unknown_location"

    # Get the robot's current position
    robot_position = hospital_map.get("nurse_station")  # Assuming starting at nurse station
    target_position = hospital_map[location]

    print(f"Planning path from {robot_position} to {target_position}...")

    # Simulate path planning
    path = np.linspace(robot_position, target_position, num=10)

    print("Path planned. Executing motion...")

    # Visualize the path (simulation)
    for point in path:
        print(f"Moving to {point}...")
        time.sleep(0.5)

    if location == "closet_bandage_bin":
        # Simulate encountering a locked door
        print("Checking if the door to the closet is locked...")
        door_locked = randint(0, 1) == 1  # Randomly determine if the door is locked
        if door_locked:
            print("The door to the closet is locked.")
            return "locked"
        else:
            print("The door to the closet is unlocked. Proceeding inside.")
    return "success"

def activate_led():
    """Simulates activating a large LED to alert staff."""
    print("[LED Activated] Large LED on top is lit to alert staff.")
    time.sleep(1)  # Simulate LED activation time

def return_to_nurse_station():
    """Simulates the robot returning to the nurse station."""
    print("Returning to the nurse station...")
    move_to_location("nurse_station")

def report_issue(time_of_request, issue, instruction):
    """Reports the issue verbally and logs it."""
    message = (
        f"At {time_of_request}, I was asked by a Nurse to {instruction}. "
        f"However, {issue}."
    )
    print(f"[Robot Report] {message}")
    tts_engine = pyttsx3.init()
    tts_engine.say(message)
    tts_engine.runAndWait()

# Main execution flow
verbal_instruction = "Bring small gauze bandages to 205"
current_time = time.strftime("%I:%M %p")

# Parse instruction
action_plan = verbal_instruction_to_action(verbal_instruction)

if action_plan:
    # Step 1: Move to the bandage bin
    result = move_to_location(action_plan["bin_location"])

    if result == "locked":
        # Step 2: Handle the locked closet
        print("Encountered a locked door at the bandage bin.")
        activate_led()
        return_to_nurse_station()

        # Step 3: Report the issue
        issue = "the door to that closet was locked, so I could not fulfill the request"
        report_issue(current_time, issue, verbal_instruction)
    else:
        # Closet was accessible (success case, if needed for future expansion)
        print("Successfully accessed the bandage bin.")
else:
    print("Could not understand the instruction.")
