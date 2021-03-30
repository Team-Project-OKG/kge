import os
from collections import OrderedDict

def extend_dataset(dataset, observed_ents, observed_rels , limit):
    data_map = {}
    i = 1
    for idx, line in enumerate(dataset):
        observed_e_count = 0
        observed_r_count = 0
        for idx_el, el in enumerate(line):
            if idx_el == 1:
                if el in observed_rels:
                    observed_r_count += 1
            if isinstance(el, list):
                for e in el:
                    if e in observed_ents:
                        observed_e_count += 1
            elif el in observed_ents:
                observed_e_count += 1
        if observed_e_count > 0 or observed_r_count > 0:
            data_map[idx] = (observed_e_count + observed_r_count, line)
            for idx_, el_ in enumerate(line):
                if idx_ == 1:
                    observed_rels.add(el_)
                elif isinstance(el_, list):
                    for x in el_:
                        observed_ents.add(x)
                else:
                    observed_ents.add(el_)
            if i == limit:
                break
            else:
                i += 1
    return data_map


# Parameters:
test_set_examples = 10  # number of final test examples
limit = 10  # Max number of lines for each final validation dataset

directory = './olpbench/mapped_to_ids/'
files = [f for f in os.listdir(directory) if f.startswith("test") or f.startswith("validation")]
lookup = {}
for filename in files:
    with open(directory + filename, "r") as f:
        dataset = f.readlines()
        print("Reading lines of {} done".format(filename))
        test = [line.rstrip().split('\t') for line in dataset]
        data_int = []
        print("Length of file {}: {}".format(filename, len(test)))
        for idx, line in enumerate(test):
            row = []
            for idx_el, el in enumerate(line):
                if [el] != el.split():
                    el = el.split()
                row.append(el)
            data_int.append(row)
        lookup[filename] = data_int

test_subset = lookup["test_data.txt"][70:70+test_set_examples]
observed_entities = set()
observed_relations = set([line[1] for line in test_subset])

for line in test_subset:
    for idx, el in enumerate(line):
        if idx == 1:
            continue
        if isinstance(el, list):
            observed_entities = observed_entities.union(set(el))
            for e in el:
                observed_entities.add(e)
        else:
            observed_entities.add(el)

test_subset = {i: (0, test_subset[i]) for i in range(test_set_examples)}
print("Test set: {} observed entities, {} observed relations".format(len(observed_entities), len(observed_relations)))
data_maps = {}


remaining_files = files.copy()
remaining_files.remove("test_data.txt")
for f in remaining_files:
    if f in remaining_files:
        print("Extend dataset {}".format(f))
        data = extend_dataset(lookup[f], observed_entities, observed_relations, limit)
        data_maps[f] = OrderedDict(sorted(data.items(), key=lambda key:key[1][0], reverse=True))
        print("Test + validation set: {} observed entities, {} observed relations".format(len(observed_entities), len(observed_relations)))

# clone observed dataset -> for all training sets -> cover all observed entities and relations
success_count = 0
total_entities = set(observed_entities)
total_relations = set(observed_relations)
training_data = [f for f in os.listdir(directory) if f.startswith("train")]
for filename in training_data:
    data_maps[filename] = {}
    to_observe_e = set(observed_entities)
    to_observe_r = set(observed_relations)
    train_observed_e = set()
    train_observed_r = set()
    with open(directory + filename, "r") as f:
        for idx, line in enumerate(f):
            observed_count_e = 0
            observed_count_r = 0
            for idx_line, el in enumerate(line.rstrip().split('\t')):
                if idx_line == 1 and el in to_observe_r:
                    observed_count_r += 1
                    to_observe_r.remove(el)
                elif el in to_observe_e:
                    observed_count_e += 1
                    to_observe_e.remove(el)
            if observed_count_r > 0 or observed_count_e > 0:
                data_maps[filename][idx] = (observed_count_e + observed_count_r, line.rstrip().split('\t'))
                train_observed_r.add(line.rstrip().split('\t')[1])
                train_observed_e = train_observed_e.union({line.rstrip().split('\t')[0], line.rstrip().split('\t')[2]})

            if idx % 1000000 == 0:
                print(
                    "{} {}: Could find {}/{} entities and {}/{} relations".format(idx, filename, len(observed_entities) - len(to_observe_e),
                                                                                  len(observed_entities),
                                                                                  len(observed_relations) - len(to_observe_r),
                                                                                  len(observed_relations)))
            if len(to_observe_e) == 0 and len(to_observe_r) == 0:
                success_count += 1
                print("Success, found all elements for {}!".format(filename))
                break

        total_entities = total_entities.union(train_observed_e)
        total_relations = total_relations.union(train_observed_r)

print("Found {} total entities".format(len(total_entities)))
print("Found {} total relations".format(len(total_relations)))

data_maps["test_data.txt"] = test_subset
if success_count == 3:
    print("Writing data...")
    for f, tuple in data_maps.items():
        new_file = open(f[:-4]+".del", "w+")
        i = 1
        for idx, line in dict(tuple).items():
            for idx_el, el in enumerate(line[1]):
                if isinstance(el, list):
                    new_file.write(" ".join(el))
                else:
                    new_file.write(el)
                if idx_el + 1 < len(line[1]):
                    new_file.write("\t")
            new_file.write("\n")
        new_file.close()

entity_file = open("entities.txt", "w+")
entity_file.write("# total entities: " + str(len(total_entities)) + "\n")
for e in total_entities:
    entity_file.write(e + "\n")
entity_file.close()
relation_file = open("relations.txt", "w+")
relation_file.write("# total relations: " + str(len(total_relations)) + "\n")
for r in total_relations:
    relation_file.write(r + "\n")
relation_file.close()
print("Done")