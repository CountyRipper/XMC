import yaml
fl = open("./conf/prefix.yaml")
args = yaml.load(fl,Loader=yaml.FullLoader)
print(args.prefix_tuning)

