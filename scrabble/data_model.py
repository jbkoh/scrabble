from mongoengine import *
from .common import *

connect('plastering')

VENDOR_GIVEN_NAME = 'VendorGivenName'
BACNET_NAME = 'BACnetName'
BACNET_DESC = 'BACnetDescription'
BACNET_UNIT = 'BACnetUnit'

column_names = [VENDOR_GIVEN_NAME,
                BACNET_NAME,
                BACNET_DESC,
                ]

class RawMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    metadata = DictField()
    meta = {'strict': False}

class LabeledMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    fullparsing = DictField()
    tagsets = ListField(StringField())
    point_tagset = StringField()
    meta = {'strict': False}

class ResultHistory(Document):
    history = ListField()
    use_brick_flag = BooleanField()
    use_known_tags = BooleanField()
    sample_num_list = ListField()
    source_building_list = ListField()
    target_building = StringField(required=True)
    negative_flag = BooleanField()
    entqs = StringField()
    crfqs = StringField()
    crfalgo = StringField(default='ap')
    tagset_classifier_type = StringField()
    postfix = StringField()
    task = StringField()
    ts_flag = BooleanField()
    sequential_type = StringField()

