from BackTranslation import BackTranslation
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

trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})

def back_translation(train_orig, output_file):

    writer = open(output_file, 'w', encoding="utf8")
    lines = open(train_orig, 'r', encoding="utf8").readlines()

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        #label = parts[0]
        sentence = parts[0]
        result = trans.translate(sentence, src='en', tmp='zh-cn')
        if isinstance(result, str):
            back_t = result
        else:
            back_t = result.result_text
        print(back_t)

        writer.write( "\t" + back_t + '\n')


    writer.close()
    print("generated augmented sentences with back translation for " + train_orig + " to " + output_file + "from English to Chinese")


#main function
if __name__ == "__main__":

    #generate augmented sentences and output into a new file
    back_translation(args.input, output)
# prefix = 'E:/lcl mixup/Contrastive_summary/processed_data/csn-main/seed-2/train/'
# train_orig = prefix + 'data'
# output_file = prefix + 'train-backtranslation'
#
# print(back_translation(train_orig, output_file))
