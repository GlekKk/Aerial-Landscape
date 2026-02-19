import splitfolders

input_folder = "./data/raw"
output_folder = "./data/processed"

splitfolders.ratio(input = input_folder, output = output_folder, 
                   seed = 42, 
                   ratio = (.7, .15, .15),
                   group_prefix = None,
                   move = False)

