import matplotlib.pyplot as plt
import numpy as np

gyro_data = [
    0.0007,
    0.0025,
    0.0017,
    0.0015,
    0.0015,
    0.0020,
    0.0039,
    0.0043,
    0.0045,
    0.0065,
    0.0076,
    0.0075,
    0.0082,
    0.0086,
    0.0092,
    0.0102,
    0.0112,
    0.0120,
    0.0129,
    0.0136,
    0.0133,
    0.0139,
    0.0150,
    0.0156,
    0.0170,
    0.0172,
    0.0184,
    0.0183,
    0.0187,
    0.0189,
    0.0194,
    0.0205,
    0.0208,
    0.0217,
    0.0219,
    0.0242,
    0.0235,
    0.0232,
    0.0230,
    0.0245,
    0.0254,
    0.0258,
    0.0252,
    0.0259,
    0.0258,
    0.0257,
    0.0258,
    0.0267,
    0.0260,
    0.0240,
    0.0228,
    0.0203,
    0.0220,
    0.0225,
    0.0218,
    0.0238,
    0.0237,
    0.0245,
    0.0251,
    0.0262,
    0.0271,
    0.0263,
    0.0281,
    0.0293,
    0.0306,
    0.0313,
    0.0328,
    0.0323,
    0.0335,
    0.0358,
    0.0350,
    0.0355,
    0.0364,
    0.0359,
    0.0360,
    0.0360,
    0.0367,
    0.0381,
    0.0388,
    0.0393,
    0.0399,
    0.0389,
    0.0390,
    0.0392,
    0.0406,
    0.0412,
    0.0421,
    0.0428,
    0.0436,
    0.0444,
    0.0447,
    0.0419,
    0.0422,
    0.0436,
    0.0442,
    0.0442,
    0.0458,
    0.0468,
    0.0469,
    0.0488,
    0.0470,
    0.0452,
    0.0467,
    0.0448,
    0.0451,
    0.0438,
    0.0448,
    0.0456,
    0.0455,
    0.0463,
    0.0461,
    0.0454,
    0.0468,
    0.0470,
    0.0474,
    0.0481,
    0.0492,
    0.0529,
    0.0491,
    0.0544,
    0.0550,
    0.0565,
    0.0558,
    0.0572,
    0.0584,
    0.0592,
    0.0601,
    0.0605,
    0.0613,
    0.0609,
    0.0611,
    0.0625,
    0.0631,
    0.0632,
    0.0633,
    0.0654,
    0.0646,
    0.0649,
    0.0660,
    0.0665,
    0.0658,
    0.0670,
    0.0680,
    0.0687,
    0.0687,
    0.0699,
    0.0690,
    0.0701,
    0.0714,
    0.0719,
    0.0708,
    0.0719,
    0.0734,
    0.0740,
    0.0743,
    0.0759,
    0.0757,
    0.0770,
    0.0774,
    0.0783,
    0.0790,
    0.0791,
    0.0792,
    0.0793,
    0.0794,
    0.0806,
    0.0809,
    0.0813,
    0.0807,
    0.0824,
    0.0832,
    0.0834,
    0.0840,
    0.0844,
    0.0848,
    0.0844,
    0.0854,
    0.0858,
    0.0868,
    0.0860,
    0.0874,
    0.0879,
    0.0878,
    0.0907,
    0.0909,
    0.0924,
    0.0918,
    0.0918,
    0.0929,
    0.0942,
    0.0939,
    0.0936,
    0.0957,
    0.0961,
    0.0979,
    0.0977,
    0.0995,
    0.0995,
    0.0979,
    0.0957,
    0.0934,
    0.0947,
    0.0952,
    0.0949,
    0.0921,
    0.0888,
    0.0918,
    0.0948,
    0.0953,
    0.0950,
    0.0945,
    0.0965,
    0.0951,
    0.0951,
    0.0961,
    0.0974,
    0.0983,
    0.0985,
    0.0985,
    0.0976,
    0.0976,
    0.0982,
    0.1001,
    0.1013,
    0.1018,
    0.1019,
    0.1018,
    0.1017,
    0.1027,
    0.1030,
    0.1054,
    0.1059,
    0.1056,
    0.1054,
    0.1054,
    0.1060,
    0.1067,
    0.1074,
    0.1083,
    0.1090,
    0.1094,
    0.1100,
    0.1106,
    0.1105,
    0.1108,
    0.1118,
    0.1128,
    0.1137,
    0.1145,
    0.1161,
    0.1128,
    0.1154,
    0.1161,
    0.1160,
    0.1169,
    0.1165,
    0.1181,
    0.1185,
    0.1189,
    0.1198,
    0.1204,
    0.1205,
    0.1214,
    0.1221,
    0.1224,
    0.1223,
    0.1233,
    0.1241,
    0.1254,
    0.1255,
    0.1269,
    0.1275,
    0.1274,
    0.1281,
    0.1281,
    0.1289,
    0.1282,
    0.1287,
    0.1296,
    0.1303,
    0.1303,
    0.1310,
    0.1317,
    0.1324,
    0.1331,
    0.1342,
    0.1342,
    0.1340,
    0.1339,
    0.1346,
    0.1357,
    0.1362,
    0.1367,
    0.1381,
    0.1382,
    0.1383,
    0.1389,
    0.1391,
    0.1398,
    0.1413,
    0.1422,
    0.1425,
    0.1428,
    0.1432,
    0.1450,
    0.1453,
    0.1464,
    0.1475,
    0.1472,
    0.1482,
    0.1487,
    0.1495,
    0.1506,
    0.1508,
    0.1513,
    0.1522,
    0.1528,
    0.1535,
    0.1539,
    0.1548,
    0.1558,
    0.1564,
    0.1576,
    0.1568,
    0.1562,
    0.1574,
    0.1579,
    0.1582,
    0.1589,
    0.1593,
    0.1593,
    0.1603,
    0.1615,
    0.1623,
    0.1634,
    0.1642,
    0.1640,
    0.1645,
    0.1655,
    0.1662,
    0.1668,
    0.1683,
    0.1686,
    0.1685,
    0.1694,
    0.1702,
    0.1711,
    0.1724,
    0.1735,
    0.1733,
    0.1733,
    0.1735,
    0.1740,
    0.1749,
    0.1760,
    0.1770,
    0.1777,
    0.1774,
    0.1776,
    0.1787,
    0.1795,
    0.1808,
    0.1814,
    0.1819,
    0.1822,
    0.1840,
    0.1825,
    0.1838,
    0.1844,
    0.1854,
    0.1853,
    0.1853,
    0.1862,
    0.1869,
    0.1878,
    0.1886,
    0.1893,
    0.1900,
    0.1914,
    0.1918,
    0.1926,
    0.1943,
    0.1960,
    0.1964,
    0.1960,
    0.1955,
    0.1958,
    0.1976,
    0.1957,
    0.1948,
    0.1993,
    0.1960,
    0.1955,
    0.1934,
    0.1901,
    0.1946,
    0.2069,
    0.1976,
    0.1832,
    0.1770,
    0.1654
]

accel_data = [
    0.0285,
    0.0219,
    0.0096,
    0.0179,
    0.0139,
    0.0096,
    0.0091,
    0.0122,
    0.0107,
    0.0085,
    0.0090,
    0.0087,
    0.0092,
    0.0102,
    0.0088,
    0.0111,
    0.0086,
    0.0094,
    0.0042,
    0.0120,
    0.0113,
    0.0147,
    0.0134,
    0.0101,
    0.0082,
    0.0140,
    0.0122,
    0.0166,
    0.0076,
    0.0102,
    0.0149,
    0.0066,
    0.0128,
    0.0119,
    0.0052,
    0.0087,
    0.0042,
    0.0026,
    0.0027,
    0.0123,
    0.0047,
    0.0024,
    0.0024,
    0.0093,
    0.0033,
    0.0015,
    0.0121,
    0.0020,
    0.0126,
    0.0012,
    0.0107,
    0.0022,
    0.0042,
    0.0116,
    0.0053,
    0.0009,
    0.0019,
    0.0024,
    0.0100,
    0.0052,
    0.0046,
    0.0072,
    0.0036,
    0.0055,
    0.0009,
    0.0044,
    0.0032,
    0.0042,
    0.0027,
    0.0095,
    0.0034,
    0.0076,
    0.0083,
    0.0092,
    0.0069,
    0.0028,
    0.0019,
    0.0100,
    0.0112,
    0.0065,
    0.0100,
    0.0029,
    0.0082,
    0.0013,
    0.0031,
    0.0014,
    0.0091,
    0.0066,
    0.0053,
    0.0139,
    0.0042,
    0.0033,
    0.0082,
    0.0032,
    0.0067,
    0.0100,
    0.0141,
    0.0075,
    0.0182,
    0.0103,
    0.0067,
    0.0153,
    0.0168,
    0.0124,
    0.0135,
    0.0163,
    0.0052,
    0.0168,
    0.0069,
    0.0073,
    0.0129,
    0.0149,
    0.0106,
    0.0123,
    0.0131,
    0.0117,
    0.0020,
    0.0154,
    0.0112,
    0.0153,
    0.0125,
    0.0149,
    0.0141,
    0.0127,
    0.0145,
    0.0138,
    0.0127,
    0.0187,
    0.0072,
    0.0089,
    0.0134,
    0.0208,
    0.0231,
    0.0157,
    0.0142,
    0.0252,
    0.0204,
    0.0158,
    0.0158,
    0.0197,
    0.0164,
    0.0173,
    0.0187,
    0.0126,
    0.0194,
    0.0142,
    0.0203,
    0.0203,
    0.0243,
    0.0149,
    0.0192,
    0.0163,
    0.0183,
    0.0180,
    0.0186,
    0.0188,
    0.0157,
    0.0178,
    0.0166,
    0.0136,
    0.0162,
    0.0212,
    0.0180,
    0.0252,
    0.0241,
    0.0215,
    0.0249,
    0.0203,
    0.0207,
    0.0112,
    0.0269,
    0.0251,
    0.0161,
    0.0076,
    0.0050,
    0.0062,
    0.0051,
    0.0085,
    0.0078,
    0.0043,
    0.0074,
    0.0030,
    0.0056,
    0.0161,
    0.0114,
    0.0146,
    0.0033,
    0.0029,
    0.0013,
    0.0069,
    0.0063,
    0.0035,
    0.0025,
    0.0088,
    0.0091,
    0.0065,
    0.0084,
    0.0109,
    0.0071,
    0.0050,
    0.0062,
    0.0095,
    0.0047,
    0.0059,
    0.0055,
    0.0126,
    0.0099,
    0.0038,
    0.0071,
    0.0109,
    0.0087,
    0.0076,
    0.0145,
    0.0107,
    0.0065,
    0.0175,
    0.0176,
    0.0112,
    0.0114,
    0.0162,
    0.0086,
    0.0054,
    0.0122,
    0.0128,
    0.0139,
    0.0121,
    0.0104,
    0.0120,
    0.0108,
    0.0113,
    0.0165,
    0.0175,
    0.0218,
    0.0242,
    0.0106,
    0.0146,
    0.0181,
    0.0203,
    0.0298,
    0.0182,
    0.0174,
    0.0232,
    0.0212,
    0.0276,
    0.0271,
    0.0196,
    0.0213,
    0.0204,
    0.0184,
    0.0099,
    0.0084,
    0.0157,
    0.0143,
    0.0197,
    0.0150,
    0.0062,
    0.0101,
    0.0110,
    0.0067,
    0.0044,
    0.0051,
    0.0019,
    0.0170,
    0.0075,
    0.0062,
    0.0031,
    0.0035,
    0.0079,
    0.0024,
    0.0049,
    0.0061,
    0.0030,
    0.0072,
    0.0091,
    0.0035,
    0.0063,
    0.0064,
    0.0042,
    0.0048,
    0.0048,
    0.0075,
    0.0075,
    0.0038,
    0.0048,
    0.0043,
    0.0063,
    0.0118,
    0.0048,
    0.0050,
    0.0080,
    0.0102,
    0.0075,
    0.0064,
    0.0057,
    0.0075,
    0.0079,
    0.0068,
    0.0114,
    0.0081,
    0.0098,
    0.0132,
    0.0094,
    0.0035,
    0.0080,
    0.0116,
    0.0132,
    0.0052,
    0.0080,
    0.0061,
    0.0120,
    0.0079,
    0.0065,
    0.0131,
    0.0115,
    0.0129,
    0.0127,
    0.0153,
    0.0102,
    0.0156,
    0.0119,
    0.0123,
    0.0203,
    0.0079,
    0.0121,
    0.0139,
    0.0074,
    0.0448,
]

joint_data = [
    0.0056,
    0.0071,
    0.0049,
    0.0134,
    0.0138,
    0.0167,
    0.0169,
    0.0135,
    0.0095,
    0.0079,
    0.0066,
    0.0068,
    0.0098,
    0.0073,
    0.0071,
    0.0056,
    0.0021,
    0.0029,
    0.0026,
    0.0029,
    0.0024,
    0.0027,
    0.0027,
    0.0021,
    0.0018,
    0.0026,
    0.0043,
    0.0047,
    0.0060,
    0.0055,
    0.0045,
    0.0062,
    0.0077,
    0.0091,
    0.0100,
    0.0107,
    0.0110,
    0.0107,
    0.0112,
    0.0117,
    0.0126,
    0.0134,
    0.0147,
    0.0145,
    0.0134,
    0.0141,
    0.0169,
    0.0151,
    0.0137,
    0.0126,
    0.0120,
    0.0113,
    0.0094,
    0.0084,
    0.0090,
    0.0082,
    0.0086,
    0.0080,
    0.0073,
    0.0072,
    0.0071,
    0.0076,
    0.0073,
    0.0073,
    0.0069,
    0.0047,
    0.0042,
    0.0053,
    0.0064,
    0.0073,
    0.0081,
    0.0082,
    0.0080,
    0.0063,
    0.0048,
    0.0054,
    0.0061,
    0.0062,
    0.0057,
    0.0056,
    0.0049,
    0.0042,
    0.0020,
    0.0023,
    0.0019,
    0.0012,
    0.0012,
    0.0024,
    0.0029,
    0.0026,
    0.0026,
    0.0030,
    0.0037,
    0.0043,
    0.0048,
    0.0053,
    0.0045,
    0.0030,
    0.0032,
    0.0033,
    0.0041,
    0.0046,
    0.0044,
    0.0041,
    0.0028,
    0.0034,
    0.0036,
    0.0041,
    0.0048,
    0.0049,
    0.0041,
    0.0037,
    0.0040,
    0.0046,
    0.0051,
    0.0039,
    0.0028,
    0.0019,
    0.0016,
    0.0021,
    0.0017,
    0.0026,
    0.0043,
    0.0041,
    0.0042,
    0.0033,
    0.0027,
    0.0036,
    0.0041,
    0.0051,
    0.0047,
    0.0031,
    0.0019,
    0.0022,
    0.0032,
    0.0042,
    0.0036,
    0.0045,
    0.0048,
    0.0046,
    0.0039,
    0.0047,
    0.0054,
    0.0065,
    0.0060,
    0.0055,
    0.0041,
    0.0035,
    0.0021,
    0.0004,
    0.0016,
    0.0028,
    0.0037,
    0.0045,
    0.0043,
    0.0029,
    0.0014,
    0.0014,
    0.0025,
    0.0029,
    0.0039,
    0.0049,
    0.0045,
    0.0017,
    0.0012,
    0.0018,
    0.0007,
    0.0010,
    0.0014,
    0.0003,
    0.0006,
    0.0010,
    0.0020,
    0.0025,
    0.0032,
    0.0026,
    0.0016,
    0.0016,
    0.0016,
    0.0022,
    0.0034,
    0.0030,
    0.0025,
    0.0022,
    0.0027,
    0.0037,
    0.0044,
    0.0043,
    0.0042,
    0.0031,
    0.0024,
    0.0020,
    0.0026,
    0.0035,
    0.0053,
    0.0064,
    0.0069,
    0.0071,
    0.0071,
    0.0058,
    0.0043,
    0.0043,
    0.0051,
    0.0061,
    0.0062,
    0.0067,
    0.0059,
    0.0048,
    0.0041,
    0.0053,
    0.0068,
    0.0068,
    0.0063,
    0.0056,
    0.0047,
    0.0043,
    0.0050,
    0.0073,
    0.0104,
    0.0095,
    0.0055,
    0.0039,
    0.0021,
    0.0019,
    0.0027,
    0.0020,
    0.0030,
    0.0033,
    0.0029,
    0.0032,
    0.0036,
    0.0038,
    0.0039,
    0.0050,
    0.0058,
    0.0048,
    0.0049,
    0.0049,
    0.0066,
    0.0081,
    0.0068,
    0.0041,
    0.0028,
    0.0039,
    0.0038,
    0.0038,
    0.0038,
    0.0041,
    0.0048,
    0.0045,
    0.0046,
    0.0050,
    0.0058,
    0.0061,
    0.0061,
    0.0059,
    0.0062,
    0.0065,
    0.0063,
    0.0065,
    0.0070,
    0.0059,
    0.0060,
    0.0061,
    0.0059,
    0.0063,
    0.0070,
    0.0072,
    0.0068,
    0.0068,
    0.0066,
    0.0068,
    0.0069,
    0.0069,
    0.0077,
    0.0072,
    0.0070,
    0.0074,
    0.0074,
    0.0076,
    0.0081,
    0.0084,
    0.0076,
    0.0070,
    0.0072,
    0.0075,
    0.0078,
    0.0069,
    0.0066,
    0.0063,
    0.0054,
    0.0047,
    0.0045,
    0.0039,
    0.0032,
    0.0033,
    0.0044,
    0.0034,
    0.0037,
    0.0029,
    0.0030,
    0.0038,
    0.0037,
    0.0054,
    0.0042,
    0.0036,
    0.0034,
    0.0034,
    0.0029,
    0.0030,
    0.0032,
    0.0042,
    0.0040,
    0.0035,
    0.0035,
    0.0033,
    0.0033,
    0.0029,
    0.0032,
    0.0027,
    0.0031,
    0.0026,
    0.0026,
    0.0032,
    0.0032,
    0.0048,
    0.0047,
    0.0034,
    0.0027,
    0.0029,
    0.0031,
    0.0036,
    0.0043,
    0.0036,
    0.0031,
    0.0028,
    0.0025,
    0.0028,
    0.0031,
    0.0043,
    0.0032,
    0.0028,
    0.0028,
    0.0029,
    0.0042,
    0.0035
]

x_values = np.array(range(5 * 60)) / 60
gyro_data = np.array(gyro_data)[10:5 * 60 + 10] * 180 / np.pi
plt.plot(x_values, gyro_data, label="Gyroscope Only")

accel_data = np.array(accel_data)[10:5 * 60 + 10] * 180 / np.pi
plt.plot(x_values, accel_data, label="Accelerometer Only")

joint_data = np.array(joint_data)[46:5 * 60 + 46] * 180 / np.pi
plt.plot(x_values, joint_data, label="Complimentary Filter")

plt.xlabel('Time (minutes)');
plt.ylabel('Tilt (degrees)');
plt.title('Tilt Comparison');

plt.legend()


plt.show()





