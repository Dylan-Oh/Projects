import yaml

data = {
        'path' : './datasets/SMD_Plus',
        'train' : 'train',
        'val' : 'val',
        'nc' : 7,
        'names' : ['Ferry', 'Buoy', 'Vessel_ship', 'Boat', 'Kayak', 'Sail_boat', 'Other']}

with open('data.yaml', 'w') as f :
    yaml.dump(data, f)