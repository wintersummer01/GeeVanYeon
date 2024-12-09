import yaml

def dumpyaml(args, save_path):
    stream = yaml.dump(args)
    stream = stream.split('\n')
    for line in stream:
        if '!!' in line:
            stream.remove(line)
    stream = '\n'.join(stream)
    
    with open(f'{save_path}/config.yaml', 'w') as f:
        f.write(stream)