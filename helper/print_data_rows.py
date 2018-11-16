import pdb
import pprint

from data_model import *
from common import *

pp = pprint.PrettyPrinter()

argparser.add_argument('-b', type=str, dest='building')
args = argparser.parse_args()
building = args.building


def print_row_by_row(building, metadata_types=['VendorGivenName']):
    srcids = [obj.srcid for obj in LabeledMetadata.objects(building=building)]
    for srcid in srcids:
        labeled = LabeledMetadata.objects(srcid=srcid).first()
        fullparsing = labeled.fullparsing
        tagsets = labeled.tagsets
        sentence = {
            metadata_type: ''.join([pair[0] for pair
                                    in fullparsing[metadata_type]])
            for metadata_type in metadata_types
        }
        print('\n')
        print('-------------------------')
        print('Sentences: ')
        pp.pprint(sentence)
        print('\n')
        print('Parsed: ')
        for metadata_type in metadata_types:
            print(metadata_type)
            parsed = fullparsing[metadata_type]
            for row in parsed:
                print(row)
        print('\n')
        print('Tagsets: ')
        pp.pprint(tagsets)
        print('\n')
        pdb.set_trace()

if __name__ == '__main__':
    print_row_by_row(building)

