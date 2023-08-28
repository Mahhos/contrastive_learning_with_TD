
import nltk
nltk.download('averaged_perceptron_tagger')
import nlpaug.augmenter.word as naw

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")

args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))


aug = naw.SynonymAug(aug_src='wordnet')


def gen_eda(train_orig, output_file):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        #label = parts[0]
        sentence = parts[0]
        # print(sentence)
        aug_sentences = aug.augment(sentence)

        writer.write( "\t" + aug_sentences + '\n')
        print(aug_sentences)


    writer.close()
    print("generated augmented sentences with synonym replacement for " + train_orig + " to " + output_file )


#main function
if __name__ == "__main__":

    #generate augmented sentences and output into a new file
    gen_eda(args.input, output)


# prefix = 'E:/lcl mixup/Contrastive_summary/processed_data/csn-main/seed-2/train/'
# train_orig = prefix + 'data'
# output_file = prefix + 'train-syn'
#
# print(gen_eda(train_orig, output_file))

