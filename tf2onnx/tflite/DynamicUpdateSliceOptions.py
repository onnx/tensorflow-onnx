# SPDX-License-Identifier: Apache-2.0

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class DynamicUpdateSliceOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDynamicUpdateSliceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DynamicUpdateSliceOptions()
        x.Init(buf, n + offset)
        return x

    # DynamicUpdateSliceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def DynamicUpdateSliceOptionsStart(builder): builder.StartObject(0)
def DynamicUpdateSliceOptionsEnd(builder): return builder.EndObject()