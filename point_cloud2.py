00001 #!/usr/bin/env python
00002
00003 # Software License Agreement (BSD License)
00004 #
00005 # Copyright (c) 208, Willow Garage, Inc.
00006 # All rights reserved.
00007 #
00008 # Redistribution and use in source and binary forms, with or without
00009 # modification, are permitted provided that the following conditions
00010 # are met:
00011 #
00012 # * Redistributions of source code must retain the above copyright
00013 # notice, this list of conditions and the following disclaimer.
00014 # * Redistributions in binary form must reproduce the above
00015 # copyright notice, this list of conditions and the following
00016 # disclaimer in the documentation and/or other materials provided
00017 # with the distribution.
00018 # * Neither the name of Willow Garage, Inc. nor the names of its
00019 # contributors may be used to endorse or promote products derived
00020 # from this software without specific prior written permission.
00021 #
00022 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
00023 # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
00024 # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
00025 # FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
00026 # COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
00027 # INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
00028 # BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
00029 # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
00030 # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
00031 # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
00032 # ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
00033 # POSSIBILITY OF SUCH DAMAGE.
00034
00035 from __future__ import print_function
00036
00037 """
00038 Serialization of sensor_msgs.PointCloud2 messages.
00039 
00040 Author: Tim Field
00041 """
00042
00043 import ctypes
00044 import math
00045 import struct
00046
00047 import roslib.message
00048 from sensor_msgs.msg import PointCloud2, PointField
00049
00050 _DATATYPES = {}
00051 _DATATYPES[PointField.INT8]    = ('b', 1)
00052 _DATATYPES[PointField.UINT8]   = ('B', 1)
00053 _DATATYPES[PointField.INT16]   = ('h', 2)
00054 _DATATYPES[PointField.UINT16]  = ('H', 2)
00055 _DATATYPES[PointField.INT32]   = ('i', 4)
00056 _DATATYPES[PointField.UINT32]  = ('I', 4)
00057 _DATATYPES[PointField.FLOAT32] = ('f', 4)
00058 _DATATYPES[PointField.FLOAT64] = ('d', 8)
00059
00060 def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
00061     """
00062     Read points from a L{sensor_msgs.PointCloud2} message.
00063 
00064     @param cloud: The point cloud to read from.
00065     @type  cloud: L{sensor_msgs.PointCloud2}
00066     @param field_names: The names of fields to read. If None, read all fields. [default: None]
00067     @type  field_names: iterable
00068     @param skip_nans: If True, then don't return any point with a NaN value.
00069     @type  skip_nans: bool [default: False]
00070     @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
00071     @type  uvs: iterable
00072     @return: Generator which yields a list of values for each point.
00073     @rtype:  generator
00074     """
00075     assert isinstance(cloud, roslib.message.Message) and cloud._type == 'sensor_msgs/PointCloud2', 'cloud is not a sensor_msgs.msg.PointCloud2'
00076     fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
00077     width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
00078     unpack_from = struct.Struct(fmt).unpack_from
00079
00080     if skip_nans:
00081         if uvs:
00082             for u, v in uvs:
00083                 p = unpack_from(data, (row_step * v) + (point_step * u))
00084                 has_nan = False
00085                 for pv in p:
00086                     if isnan(pv):
00087                         has_nan = True
00088                         break
00089                 if not has_nan:
00090                     yield p
00091         else:
00092             for v in range(height):
00093                 offset = row_step * v
00094                 for u in range(width):
00095                     p = unpack_from(data, offset)
00096                     has_nan = False
00097                     for pv in p:
00098                         if isnan(pv):
00099                             has_nan = True
00100                             break
00101                     if not has_nan:
00102                         yield p
00103                     offset += point_step
00104     else:
00105         if uvs:
00106             for u, v in uvs:
00107                 yield unpack_from(data, (row_step * v) + (point_step * u))
00108         else:
00109             for v in range(height):
00110                 offset = row_step * v
00111                 for u in range(width):
00112                     yield unpack_from(data, offset)
00113                     offset += point_step
00114
00115 def create_cloud(header, fields, points):
00116     """
00117     Create a L{sensor_msgs.msg.PointCloud2} message.
00118 
00119     @param header: The point cloud header.
00120     @type  header: L{std_msgs.msg.Header}
00121     @param fields: The point cloud fields.
00122     @type  fields: iterable of L{sensor_msgs.msg.PointField}
00123     @param points: The point cloud points.
00124     @type  points: list of iterables, i.e. one iterable for each point, with the
00125                    elements of each iterable being the values of the fields for 
00126                    that point (in the same order as the fields parameter)
00127     @return: The point cloud.
00128     @rtype:  L{sensor_msgs.msg.PointCloud2}
00129     """
00130
00131     cloud_struct = struct.Struct(_get_struct_fmt(False, fields))
00132
00133     buff = ctypes.create_string_buffer(cloud_struct.size * len(points))
00134
00135     point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
00136     offset = 0
00137     for p in points:
00138         pack_into(buff, offset, *p)
00139         offset += point_step
00140
00141     return PointCloud2(header=header,
00142                        height=1,
00143                        width=len(points),
00144                        is_dense=False,
00145                        is_bigendian=False,
00146                        fields=fields,
00147                        point_step=cloud_struct.size,
00148                        row_step=cloud_struct.size * len(points),
00149                        data=buff.raw)
00150
00151 def create_cloud_xyz32(header, points):
00152     """
00153     Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).
00154 
00155     @param header: The point cloud header.
00156     @type  header: L{std_msgs.msg.Header}
00157     @param points: The point cloud points.
00158     @type  points: iterable
00159     @return: The point cloud.
00160     @rtype:  L{sensor_msgs.msg.PointCloud2}
00161     """
00162     fields = [PointField('x', 0, PointField.FLOAT32, 1),
00163               PointField('y', 4, PointField.FLOAT32, 1),
00164               PointField('z', 8, PointField.FLOAT32, 1)]
00165     return create_cloud(header, fields, points)
00166
00167 def _get_struct_fmt(is_bigendian, fields, field_names=None):
00168     fmt = '>' if is_bigendian else '<'
00169
00170     offset = 0
00171     for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
00172         if offset < field.offset:
00173             fmt += 'x' * (field.offset - offset)
00174             offset = field.offset
00175         if field.datatype not in _DATATYPES:
00176             print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
00177         else:
00178             datatype_fmt, datatype_length = _DATATYPES[field.datatype]
00179             fmt    += field.count * datatype_fmt
00180             offset += field.count * datatype_length
00181
00182     return fmt