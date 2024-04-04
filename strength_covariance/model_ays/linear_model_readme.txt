data import:
Initial shape: (163, 82)
initial factor list = 66
removed 4 very sparse columns: ['bulk_modulus_diamond', 'c44_diamond', 'c12_diamond', 'c11_diamond']
removed 23 empty columns: ['thermal_expansion_coeff_bcc', 'thermal_expansion_coeff_diamond', 'surface_energy_100_bcc', 'surface_energy_110_bcc', 'surface_energy_111_bcc', 'surface_energy_121_bcc', 'ideal_surface_energy_100_bcc', 'ideal_surface_energy_110_bcc', 'ideal_surface_energy_111_bcc', 'ideal_surface_energy_121_bcc', 'relaxed_formation_potential_energy_bcc', 'relaxed_formation_potential_energy_diamond', 'relaxed_formation_potential_energy_hcp', 'unrelaxed_formation_potential_energy_bcc', 'unrelaxed_formation_potential_energy_diamond', 'unrelaxed_formation_potential_energy_hcp', 'vacancy_migration_energy_bcc', 'vacancy_migration_energy_diamond', 'vacancy_migration_energy_hcp', 'vacancy_migration_energy_sc', 'relaxation_volume_bcc', 'relaxation_volume_diamond', 'relaxation_volume_hcp']
cleaned data
final factor count is 39: ['lattice_constant_bcc', 'lattice_constant_diamond', 'lattice_constant_fcc', 'lattice_constant_sc', 'bulk_modulus_bcc', 'bulk_modulus_fcc', 'bulk_modulus_sc', 'c44_bcc', 'c44_fcc', 'c44_sc', 'c12_bcc', 'c12_fcc', 'c12_sc', 'c11_bcc', 'c11_fcc', 'c11_sc', 'cohesive_energy_bcc', 'cohesive_energy_diamond', 'cohesive_energy_fcc', 'cohesive_energy_sc', 'thermal_expansion_coeff_fcc', 'surface_energy_100_fcc', 'surface_energy_110_fcc', 'surface_energy_111_fcc', 'surface_energy_121_fcc', 'ideal_surface_energy_100_fcc', 'ideal_surface_energy_110_fcc', 'ideal_surface_energy_111_fcc', 'ideal_surface_energy_121_fcc', 'extr_stack_fault_energy_fcc', 'intr_stack_fault_energy_fcc', 'unstable_stack_energy_fcc', 'unstable_stack_energy_slip_fraction_fcc', 'unstable_twinning_energy_fcc', 'unstable_twinning_energy_slip_fraction_fcc', 'relaxed_formation_potential_energy_fcc', 'unrelaxed_formation_potential_energy_fcc', 'vacancy_migration_energy_fcc', 'relaxation_volume_fcc']
shuffled


--------linear model----------
all factor linear model does not include vacancy formation/migration energies at this point in manuscript
35 factors: Index(['c12_bcc', 'unstable_stack_energy_slip_fraction_fcc',
       'surface_energy_110_fcc', 'bulk_modulus_fcc',
       'ideal_surface_energy_100_fcc', 'lattice_constant_bcc', 'c44_sc',
       'cohesive_energy_sc', 'lattice_constant_diamond', 'bulk_modulus_bcc',
       'ideal_surface_energy_110_fcc', 'c12_sc', 'cohesive_energy_bcc',
       'unstable_stack_energy_fcc', 'cohesive_energy_fcc', 'c11_fcc',
       'unstable_twinning_energy_fcc', 'c11_sc', 'thermal_expansion_coeff_fcc',
       'c44_bcc', 'unstable_twinning_energy_slip_fraction_fcc', 'c12_fcc',
       'ideal_surface_energy_111_fcc', 'lattice_constant_sc',
       'extr_stack_fault_energy_fcc', 'surface_energy_100_fcc', 'c11_bcc',
       'intr_stack_fault_energy_fcc', 'ideal_surface_energy_121_fcc',
       'surface_energy_121_fcc', 'lattice_constant_fcc', 'c44_fcc',
       'surface_energy_111_fcc', 'bulk_modulus_sc', 'cohesive_energy_diamond'],
      dtype='object')

corr_list = ['c44_fcc', 'lattice_constant_bcc', 'lattice_constant_fcc', 'c11_fcc', 'surface_energy_110_fcc', 'surface_energy_100_fcc', 'ideal_surface_energy_100_fcc', 'surface_energy_121_fcc', 'lattice_constant_sc', 'ideal_surface_energy_110_fcc', 'cohesive_energy_fcc', 'ideal_surface_energy_121_fcc', 'unstable_twinning_energy_fcc', 'c11_sc', 'surface_energy_111_fcc', 'unstable_stack_energy_fcc', 'ideal_surface_energy_111_fcc', 'cohesive_energy_bcc', 'bulk_modulus_fcc', 'lattice_constant_diamond', 'cohesive_energy_sc', 'c12_fcc', 'c44_bcc', 'bulk_modulus_sc', 'thermal_expansion_coeff_fcc', 'cohesive_energy_diamond', 'c44_sc', 'c12_sc', 'intr_stack_fault_energy_fcc', 'unstable_twinning_energy_slip_fraction_fcc', 'unstable_stack_energy_slip_fraction_fcc', 'c12_bcc', 'extr_stack_fault_energy_fcc', 'bulk_modulus_bcc', 'c11_bcc']

35 all factors model: Index(['c44_fcc', 'lattice_constant_bcc', 'lattice_constant_fcc', 'c11_fcc',
       'surface_energy_110_fcc', 'surface_energy_100_fcc',
       'ideal_surface_energy_100_fcc', 'surface_energy_121_fcc',
       'lattice_constant_sc', 'ideal_surface_energy_110_fcc',
       'cohesive_energy_fcc', 'ideal_surface_energy_121_fcc',
       'unstable_twinning_energy_fcc', 'c11_sc', 'surface_energy_111_fcc',
       'unstable_stack_energy_fcc', 'ideal_surface_energy_111_fcc',
       'cohesive_energy_bcc', 'bulk_modulus_fcc', 'lattice_constant_diamond',
       'cohesive_energy_sc', 'c12_fcc', 'c44_bcc', 'bulk_modulus_sc',
       'thermal_expansion_coeff_fcc', 'cohesive_energy_diamond', 'c44_sc',
       'c12_sc', 'intr_stack_fault_energy_fcc',
       'unstable_twinning_energy_slip_fraction_fcc',
       'unstable_stack_energy_slip_fraction_fcc', 'c12_bcc',
       'extr_stack_fault_energy_fcc', 'bulk_modulus_bcc', 'c11_bcc'],
      dtype='object')

3 factor model: Index(['vacancy_migration_energy_fcc', 'surface_energy_111_fcc',
       'lattice_constant_fcc'],
      dtype='object')


---------------------
3 factor leave-one-out results

     strength    predicted       error  rel error species
21      308.0   213.673246   94.326754   0.441453      Pb
143     309.0   213.519630   95.480370   0.447174      Pb
105     375.0   145.152995  229.847005   1.583481      Pb
71      488.0   300.812845  187.187155   0.622271      Pb
64      501.0   307.387903  193.612097   0.629862      Al
109     542.0   477.147700   64.852300   0.135917      Al
15      552.0   517.132640   34.867360   0.067424      Al
10      621.0   455.999026  165.000974   0.361845      Al
50      636.0   474.075895  161.924105   0.341557      Al
32      652.0   681.250068  -29.250068  -0.042936      Al
56      683.0   810.425714 -127.425714  -0.157233      Al
126     745.0  1003.997796 -258.997796  -0.257966      Al
76      758.0   885.799309 -127.799309  -0.144276      Al
123     764.0   541.778116  222.221884   0.410171      Au
69      789.0   925.628461 -136.628461  -0.147606      Al
115     803.0   984.892129 -181.892129  -0.184682      Au
36      804.0   995.807824 -191.807824  -0.192615      Au
14      809.0   984.252069 -175.252069  -0.178056      Al
114     811.0   870.005190  -59.005190  -0.067822      Au
80      814.0   870.557519  -56.557519  -0.064967      Au
132     817.0   925.757827 -108.757827  -0.117480      Al
127     829.0   932.252862 -103.252862  -0.110756      Au
116     832.0   937.741407 -105.741407  -0.112762      Ag
150     832.0   932.584988 -100.584988  -0.107856      Au
67      833.0  1052.145162 -219.145162  -0.208284      Au
135     836.0   932.715984  -96.715984  -0.103693      Au
31      837.0   937.427536 -100.427536  -0.107131      Ag
62      847.0   983.966146 -136.966146  -0.139198      Au
94      858.0   981.946440 -123.946440  -0.126225      Ag
23      891.0  1047.759896 -156.759896  -0.149614      Au
90      893.0   902.116805   -9.116805  -0.010106      Al
100     899.0   897.733781    1.266219   0.001410      Al
60      905.0  1014.422113 -109.422113  -0.107866      Ag
66      908.0  1014.386749 -106.386749  -0.104878      Ag
134     908.0  1014.019809 -106.019809  -0.104554      Ag
6       912.0  1013.682446 -101.682446  -0.100310      Ag
12      912.0  1013.972132 -101.972132  -0.100567      Ag
58      923.0  1014.593656  -91.593656  -0.090276      Ag
74      965.0   997.139131  -32.139131  -0.032231      Pd
103     977.0  1006.375495  -29.375495  -0.029189      Pd
128     986.0  1194.003591 -208.003591  -0.174207      Al
120     987.0  1005.813417  -18.813417  -0.018705      Pd
3       991.0  1078.625544  -87.625544  -0.081238      Au
7       998.0   926.287512   71.712488   0.077419      Al
17     1027.0   963.983171   63.016829   0.065371      Al
119    1047.0  1592.271484 -545.271484  -0.342449      Cu
104    1064.0  1153.752885  -89.752885  -0.077792      Ag
75     1064.0  1082.514826  -18.514826  -0.017104      Cu
112    1067.0  1276.344976 -209.344976  -0.164019      Pd
85     1069.0   949.564693  119.435307   0.125779      Al
29     1072.0  1237.410449 -165.410449  -0.133675      Cu
148    1075.0   949.465723  125.534277   0.132216      Al
34     1077.0  1153.985604  -76.985604  -0.066713      Ag
28     1077.0   949.433109  127.566891   0.134361      Al
145    1077.0   962.319671  114.680329   0.119171      Al
95     1082.0  1077.837940    4.162060   0.003861      Cu
122    1086.0  1082.257766    3.742234   0.003458      Cu
43     1088.0   932.029115  155.970885   0.167346      Al
22     1091.0  1078.981688   12.018312   0.011139      Cu
39     1095.0  1236.664719 -141.664719  -0.114554      Cu
81     1096.0   992.196715  103.803285   0.104620      Al
38     1097.0  1067.998920   29.001080   0.027155      Cu
41     1097.0  1078.217812   18.782188   0.017420      Cu
110    1099.0   418.689184  680.310816   1.624859      Al
11     1102.0   992.263826  109.736174   0.110592      Al
77     1107.0   992.910999  114.089001   0.114904      Al
91     1118.0   992.144213  125.855787   0.126852      Al
138    1126.0   991.873351  134.126649   0.135226      Al
68     1143.0  1236.279740  -93.279740  -0.075452      Cu
82     1143.0  1236.957237  -93.957237  -0.075958      Cu
106    1144.0  1011.869457  132.130543   0.130581      Al
87     1145.0  1236.229200  -91.229200  -0.073796      Cu
92     1149.0  1236.172604  -87.172604  -0.070518      Cu
8      1151.0  1195.245208  -44.245208  -0.037018      Cu
146    1157.0  1235.601410  -78.601410  -0.063614      Cu
101    1159.0  1194.996004  -35.996004  -0.030122      Cu
24     1160.0  1451.107817 -291.107817  -0.200611      Pt
46     1161.0  1196.553530  -35.553530  -0.029713      Cu
151    1168.0  1332.154214 -164.154214  -0.123225      Cu
49     1172.0  1232.400302  -60.400302  -0.049010      Cu
40     1189.0  1427.270967 -238.270967  -0.166942      Cu
55     1190.0  1505.438906 -315.438906  -0.209533      Cu
61     1196.0  1427.799881 -231.799881  -0.162348      Pd
124    1200.0  1194.676190    5.323810   0.004456      Cu
136    1209.0  1246.392393  -37.392393  -0.030000      Cu
33     1230.0  1391.607312 -161.607312  -0.116130      Al
25     1233.0  1525.002611 -292.002611  -0.191477      Pt
4      1248.0  1156.256979   91.743021   0.079345      Cu
65     1270.0  1523.401047 -253.401047  -0.166339      Pd
139    1280.0  1560.222978 -280.222978  -0.179604      Cu
129    1326.0  1489.252888 -163.252888  -0.109621      Ag
102    1345.0  1353.706470   -8.706470  -0.006432      Au
83     1351.0  1466.103851 -115.103851  -0.078510      Pd
111    1354.0  1486.912750 -132.912750  -0.089388      Ag
52     1370.0  1351.191605   18.808395   0.013920      Al
130    1372.0  1697.774723 -325.774723  -0.191883      Pd
59     1375.0  1357.419905   17.580095   0.012951      Au
147    1376.0  1018.954865  357.045135   0.350403      Al
26     1389.0  1538.949465 -149.949465  -0.097436      Cu
107    1400.0   924.850911  475.149089   0.513757      Cu
70     1402.0   924.813820  477.186180   0.515981      Cu
117    1454.0  1517.085416  -63.085416  -0.041583      Pt
131    1457.0  1622.747995 -165.747995  -0.102140      Cu
89     1471.0  1626.831127 -155.831127  -0.095788      Cu
96     1473.0  1627.072941 -154.072941  -0.094693      Cu
16     1482.0  1628.528980 -146.528980  -0.089976      Cu
84     1545.0  1448.061500   96.938500   0.066944      Pt
125    1550.0  1441.276306  108.723694   0.075436      Pt
37     1569.0  1527.776707   41.223293   0.026983      Cu
51     1622.0  1897.496865 -275.496865  -0.145190      Ni
149    1644.0  1897.920019 -253.920019  -0.133789      Ni
45     1748.0  1712.050158   35.949842   0.020998      Ni
98     1761.0  2004.118139 -243.118139  -0.121309      Ni
57     1773.0  1610.806404  162.193596   0.100691      Cu
47     1790.0  1669.495285  120.504715   0.072180      Ni
140    1813.0  2109.296674 -296.296674  -0.140472      Ni
72     1826.0  1666.846204  159.153796   0.095482      Ni
108    1842.0  1566.243559  275.756441   0.176062      Cu
144    1858.0  1804.006548   53.993452   0.029930      Ni
9      1875.0  1803.464965   71.535035   0.039665      Ni
53     1881.0  1798.991821   82.008179   0.045586      Ni
27     1894.0  2495.853616 -601.853616  -0.241141      Ni
86     1907.0  1798.553336  108.446664   0.060297      Ni
73     1931.0  2112.744171 -181.744171  -0.086023      Ni
42     1936.0  1508.671429  427.328571   0.283248      Ni
30     1960.0  2226.039462 -266.039462  -0.119512      Ni
48     1969.0  1685.502777  283.497223   0.168197      Ni
44     2017.0  1790.350298  226.649702   0.126595      Ni
54     2021.0  1944.890781   76.109219   0.039133      Ni
141    2056.0  1789.849639  266.150361   0.148700      Ni
113    2069.0  1944.057581  124.942419   0.064269      Ni
2      2102.0  2165.115504  -63.115504  -0.029151      Ni
20     2108.0  1767.242499  340.757501   0.192819      Ni
0      2111.0  2161.434407  -50.434407  -0.023334      Ni
63     2118.0  2160.848153  -42.848153  -0.019829      Ni
5      2121.0  2156.756098  -35.756098  -0.016579      Ni
1      2122.0  2156.739867  -34.739867  -0.016108      Ni
99     2124.0  2157.994389  -33.994389  -0.015753      Ni
137    2124.0  2164.664213  -40.664213  -0.018785      Ni
13     2150.0  2160.127155  -10.127155  -0.004688      Ni
152    2194.0  2036.536567  157.463433   0.077319      Ni
19     2215.0  1875.201628  339.798372   0.181206      Ni
133    2308.0  1953.238838  354.761162   0.181627      Ni
79     2524.0  2282.971988  241.028012   0.105576      Ni
97     2526.0  2282.823381  243.176619   0.106525      Ni
88     2526.0  2759.327309 -233.327309  -0.084559      Rh
142    2546.0  2282.233672  263.766328   0.115574      Ni
93     2550.0  2277.169218  272.830782   0.119811      Ni
121    2632.0  1995.486013  636.513987   0.318977      Ni
78     2647.0  2536.109386  110.890614   0.043725      Ni
35     2766.0  2426.537168  339.462832   0.139896      Ni
18     2896.0  2421.249789  474.750211   0.196077      Ni
118    3364.0  2948.260733  415.739267   0.141012      Rh