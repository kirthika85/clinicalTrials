import json   
    
def parse_eligibility_criteria(file_path):
    try:

        with open(file_path, 'r') as file:
            data = json.load(file)

        eligibility_criteria = data['studies'][0]['protocolSection']['eligibilityModule']['eligibilityCriteria']

        inclusion_criteria = eligibility_criteria.split("Exclusion Criteria:")[0].strip()
        exclusion_criteria = "Exclusion Criteria: \n" + eligibility_criteria.split("Exclusion Criteria:")[1].strip()

        print(inclusion_criteria)
        print("\n")
        print(exclusion_criteria)
        print("\n")

    except KeyError as e:
        print(f"KeyError: {e} not found in the JSON data.")
        return [], [], {}

# Specify the path to the JSON file
file_path = '/Users/amanulla.shaik/work/clinicaltrial/data/NCT04939766.json'

# Read and print the "test" attribute and its content
parse_eligibility_criteria(file_path)
