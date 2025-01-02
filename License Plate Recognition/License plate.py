import pandas as pd
import openpyxl


excel_file = r'E:\New folder\AI Projects\License Plate Recognition\random_vehicle_data.xlsx'

def check_add_vehicle(vehicle_number, excel_file):
    
    df = pd.read_excel(excel_file)

    if vehicle_number in df['Vehicle Number'].values:
        vehicle_info = df[df['Vehicle Number'] == vehicle_number]
        print(vehicle_info)
    else:
        name = input("Enter Name: ")
        state = input("Enter State: ")
        vehicle_no = input("Enter Vehicle Number: ")

        new_vehicle = {
            "Name": name,
            "Vehicle Number": vehicle_no,
            "State": state
        }

        new_df = pd.DataFrame([new_vehicle])
        df = pd.concat([df, new_df], ignore_index = True)

        df.to_excel(excel_file, index=False)
        print("New Vehicle Info added to dataset.")

vehicle_number = input("Enter Vehicle Number: ")
check_add_vehicle(vehicle_number, excel_file)

