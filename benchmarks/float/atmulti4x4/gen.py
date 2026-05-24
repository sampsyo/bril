from jinja2 import Template
import numpy as np

matrices = [
    [[8, 3, 2, 4],[2, 7, 4, 5],[0, 1, 2, 3],[0, 1, 2, 3]],
    [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 16]],
    [[17, 18, 19, 20],[21, 22, 23, 24],[25, 26, 27, 28],[29, 30, 31, 32]],
    [[33, 34, 35, 36],[37, 38, 39, 40],[41, 42, 43, 44],[45, 46, 47, 48]],
    [[49, 50, 51, 52],[53, 54, 55, 56],[57, 58, 59, 60],[61, 62, 63, 64]],
    [[65, 66, 67, 68],[69, 70, 71, 72],[73, 74, 75, 76],[77, 78, 79, 80]],
    [[0.12, 0.45, 0.67, 0.89], [1.23, 4.56, 7.89, 0.12], [3.45, 6.78, 9.01, 2.34], [5.67, 8.90, 1.23, 4.56]],
    [[7.12, 8.34, 9.56, 0.78], [1.91, 2.34, 5.67, 8.90], [1.12, 3.45, 6.78, 9.01], [2.34, 5.67, 8.90, 1.23]],
    [[4.56, 7.89, 0.12, 3.45], [6.78, 9.01, 2.34, 5.67], [8.90, 1.23, 4.56, 7.89], [0.12, 3.45, 6.78, 9.01]],
    [[2.34, 5.67, 8.90, 1.23], [4.56, 7.89, 0.12, 3.45], [6.78, 9.01, 2.34, 5.67], [8.90, 1.23, 4.56, 7.89]],
    [[0.12, 3.45, 6.78, 9.01], [2.34, 5.67, 8.90, 1.23], [4.56, 7.89, 0.12, 3.45], [6.78, 9.01, 2.34, 5.67]],
    [[8.90, 1.23, 4.56, 7.89], [0.12, 3.45, 6.78, 9.01], [2.34, 5.67, 8.90, 1.23], [4.56, 7.89, 0.12, 3.45]],
    [[6.78, 9.01, 2.34, 5.67], [8.90, 1.23, 4.56, 7.89], [0.12, 3.45, 6.78, 9.01], [2.34, 5.67, 8.90, 1.23]],
    [[4.56, 7.89, 0.12, 3.45], [6.78, 9.01, 2.34, 5.67], [8.90, 1.23, 4.56, 7.89], [0.12, 3.45, 6.78, 9.01]],
    [[2.34, 5.67, 8.90, 1.23], [4.56, 7.89, 0.12, 3.45], [6.78, 9.01, 2.34, 5.67], [8.90, 1.23, 4.56, 7.89]],
    [[0.12, 3.45, 6.78, 9.01], [2.34, 5.67, 8.90, 1.23], [4.56, 7.89, 0.12, 3.45], [6.78, 9.01, 2.34, 5.67]],
    [[9.87, 4.32, 1.09, 7.65], [2.98, 5.43, 8.76, 1.23], [4.56, 7.89, 0.11, 3.44], [6.77, 9.00, 2.33, 5.66]],
    [[3.21, 6.54, 9.87, 1.20], [4.53, 7.86, 0.19, 3.52], [6.85, 9.18, 2.51, 5.84], [8.17, 0.50, 4.83, 7.16]],
    [[1.49, 4.82, 8.15, 0.48], [3.81, 7.14, 0.47, 3.80], [6.13, 9.46, 2.79, 6.12], [8.45, 1.78, 5.11, 8.44]],
    [[0.77, 4.10, 7.43, 0.76], [3.09, 6.42, 9.75, 3.08], [5.41, 8.74, 2.07, 5.40], [7.73, 1.06, 4.39, 7.72]],
    [[2.05, 5.38, 8.71, 1.04], [4.37, 7.70, 0.03, 4.36], [6.69, 9.02, 2.35, 6.68], [8.01, 1.34, 4.67, 8.00]],
    [[3.33, 6.66, 9.99, 1.32], [4.65, 7.98, 1.31, 4.64], [6.97, 9.30, 2.63, 6.96], [9.29, 1.62, 4.95, 9.28]],
    [[1.61, 4.94, 8.27, 0.60], [3.93, 7.26, 0.59, 3.92], [6.25, 9.58, 2.91, 6.24], [8.57, 1.90, 5.23, 8.56]],
    [[0.89, 4.22, 7.55, 0.88], [3.21, 6.54, 9.87, 3.20], [5.53, 8.86, 2.19, 5.52], [7.85, 1.18, 4.51, 7.84]],
    [[2.17, 5.50, 8.83, 1.16], [4.49, 7.82, 1.15, 4.48], [6.81, 9.14, 2.47, 6.80], [9.13, 1.46, 4.79, 9.12]],
    [[3.45, 6.78, 0.11, 3.44], [5.77, 9.10, 2.43, 5.76], [8.09, 0.42, 4.75, 8.08], [0.41, 3.74, 7.07, 0.40]],
[
    [0.28457546132827105101, 0.88627381330549304117, 0.40320221937108335908, 0.30618816153541617009],
    [0.24348777186925174565, 0.50072784147668758514, 0.48842330783646209502, 0.19709424381285589600],
    [0.33713629192588301375, 0.91693290018374307149, 0.08621385671306092124, 0.78770454526290512032],
    [0.99145116252088472120, 0.30059001519913997047, 0.42804717767327410405, 0.64424647755355002321],
],
[
    [0.48030881806754244234, 0.90259649449860590575, 0.67413645718059722611, 0.20232363120920468513],
    [0.36622239101634967984, 0.89406270127513465251, 0.95325893823456808729, 0.06421372832690142030],
    [0.24466327506236373868, 0.79081601532482548311, 0.75953013298928062635, 0.88266074171091057909],
    [0.41575699163745793996, 0.23821703956939355162, 0.58730964538244068152, 0.93577202842527273940],
],
[
    [0.43989503735275387042, 0.86580910933685562014, 0.92210690549369112023, 0.82728956316327195708],
    [0.33625499552846527251, 0.09078152680887840997, 0.26865781289550894062, 0.85192843333728907051],
    [0.46296983660445939490, 0.05022771089161170988, 0.38981476933462932966, 0.67712015876507769541],
    [0.11654404309526489314, 0.35954340499047654500, 0.93342708656642436882, 0.46231808668802909512],
],
[
    [0.36831544802535831629, 0.53198852149162323411, 0.01728710620287032818, 0.21467245598830841935],
    [0.16196980816140538195, 0.95281720171188366564, 0.33964868821846072588, 0.47546474995776200068],
    [0.09678521232130721241, 0.77746869576090238407, 0.63808430124154613683, 0.53161913932088511459],
    [0.59829189622507294999, 0.02930590131640231286, 0.80741837880772915348, 0.92115142723527643209],
],
[
    [0.13573288328419841342, 0.24368095411816059759, 0.32119334495243034855, 0.26450568384973138780],
    [0.48174270084699799543, 0.04291930261229626176, 0.47157097095133587716, 0.35838537690657223944],
    [0.77553803969512746797, 0.92408995983327890666, 0.63204407523033578897, 0.68803668226751812931],
    [0.17434141787231605125, 0.86691470925322711150, 0.32403871366539904741, 0.54071477366476827786],
],
[
    [0.14753592794468339822, 0.37806407892282123395, 0.25802388694408040504, 0.15776691010362897671],
    [0.75505246838259265640, 0.25212892065216507831, 0.62750681288657339518, 0.16400453516603197279],
    [0.66443125157348081888, 0.40209692978091393645, 0.30590717432995662151, 0.04215241442327544164],
    [0.63819395843850812433, 0.40249830989915769131, 0.11770185928804044462, 0.64643350018143419522],
],
[
    [0.35923563112553247301, 0.68424507553533164828, 0.75062910406187488555, 0.70337448879700925630],
    [0.98750268901617577200, 0.10553122798450743913, 0.51443500172643952251, 0.68988281040887666773],
    [0.12312974174209968814, 0.12200042713941117167, 0.88343121075045694113, 0.48611329760158766833],
    [0.15648740315474041207, 0.09424294560749713057, 0.37418994981271902489, 0.35703367974888411407],
],
[
    [0.40681021366070896361, 0.78620457480967287367, 0.68495447465686920552, 0.83102289006824336948],
    [0.93123388778495164164, 0.44595250556622301197, 0.15935476842946916243, 0.17258688305065117419],
    [0.75033536303401315859, 0.32787262043919784826, 0.39514666005546550398, 0.54373254260484937816],
    [0.78765577262333230646, 0.88051572263529320761, 0.10630017474477526651, 0.73095618076293988885],
],
[
    [0.50792803247678042222, 0.56827175626472137271, 0.12678695892539348922, 0.55943568334025495226],
    [0.78925031930876643482, 0.52617888485160624334, 0.79965982298913795834, 0.35414076210837253100],
    [0.26044776325265950323, 0.23478698236881204164, 0.76698617153691495130, 0.39335203319833256241],
    [0.74837189726470898510, 0.03186591535360372429, 0.34182671149417553913, 0.61461244418348870422],
],
[
    [0.74258329256046085032, 0.41690374497873239346, 0.01987092206255436366, 0.27300830543424725594],
    [0.83326637938900682823, 0.25703433984650458921, 0.07093626648498130294, 0.91022274932270630377],
    [0.59582503579951373585, 0.66916322732387678585, 0.21309321101146705413, 0.41487854039487981339],
    [0.43132289448645944052, 0.26974375677529716100, 0.22308283356038438594, 0.59828569565852096623],
],
[
    [8.0, 3.0, 2.0, 4.0],
    [2.0, 7.0, 4.0, 5.0],
    [1.0, 1.0, 2.0, 3.0],
    [1.0, 1.0, 2.0, 3.0],
],
[
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
],
[
    [17.0, 18.0, 19.0, 20.0],
    [21.0, 22.0, 23.0, 24.0],
    [25.0, 26.0, 27.0, 28.0],
    [29.0, 30.0, 31.0, 32.0],
],
[
    [33.0, 34.0, 35.0, 36.0],
    [37.0, 38.0, 39.0, 40.0],
    [41.0, 42.0, 43.0, 44.0],
    [45.0, 46.0, 47.0, 48.0],
],
[
    [49.0, 50.0, 51.0, 52.0],
    [53.0, 54.0, 55.0, 56.0],
    [57.0, 58.0, 59.0, 60.0],
    [61.0, 62.0, 63.0, 64.0],
],
[
    [65.0, 66.0, 67.0, 68.0],
    [69.0, 70.0, 71.0, 72.0],
    [73.0, 74.0, 75.0, 76.0],
    [77.0, 78.0, 79.0, 80.0],
],
[
    [1.0, 1.0, 1.0, 1.0],
    [1.2, 4.6, 7.9, 1.0],
    [3.5, 6.8, 9.0, 2.3],
    [5.7, 8.9, 1.2, 4.6],
],
[
    [7.1, 8.3, 9.6, 1.0],
    [1.9, 2.3, 5.7, 8.9],
    [1.1, 3.5, 6.8, 9.0],
    [2.3, 5.7, 8.9, 1.2],
],
[
    [4.6, 7.9, 1.0, 3.5],
    [6.8, 9.0, 2.3, 5.7],
    [8.9, 1.2, 4.6, 7.9],
    [1.0, 3.5, 6.8, 9.0],
],
[
    [2.3, 5.7, 8.9, 1.2],
    [4.6, 7.9, 1.0, 3.5],
    [6.8, 9.0, 2.3, 5.7],
    [8.9, 1.2, 4.6, 7.9],
],
[
    [1.0, 3.5, 6.8, 9.0],
    [2.3, 5.7, 8.9, 1.2],
    [4.6, 7.9, 1.0, 3.5],
    [6.8, 9.0, 2.3, 5.7],
],
[
    [8.9, 1.2, 4.6, 7.9],
    [1.0, 3.5, 6.8, 9.0],
    [2.3, 5.7, 8.9, 1.2],
    [4.6, 7.9, 1.0, 3.5],
],
[
    [6.8, 9.0, 2.3, 5.7],
    [8.9, 1.2, 4.6, 7.9],
    [1.0, 3.5, 6.8, 9.0],
    [2.3, 5.7, 8.9, 1.2],
],
[
    [4.6, 7.9, 1.0, 3.5],
    [6.8, 9.0, 2.3, 5.7],
    [8.9, 1.2, 4.6, 7.9],
    [1.0, 3.5, 6.8, 9.0],
],
[
    [2.3, 5.7, 8.9, 1.2],
    [4.6, 7.9, 1.0, 3.5],
    [6.8, 9.0, 2.3, 5.7],
    [8.9, 1.2, 4.6, 7.9],
],
[
    [1.0, 3.5, 6.8, 9.0],
    [2.3, 5.7, 8.9, 1.2],
    [4.6, 7.9, 1.0, 3.5],
    [6.8, 9.0, 2.3, 5.7],
],
[
    [9.9, 4.3, 1.1, 7.7],
    [3.0, 5.4, 8.8, 1.2],
    [4.6, 7.9, 1.0, 3.4],
    [6.8, 9.0, 2.3, 5.7],
],
[
    [3.2, 6.5, 9.9, 1.2],
    [4.5, 7.9, 1.0, 3.5],
    [6.8, 9.2, 2.5, 5.8],
    [8.2, 1.0, 4.8, 7.2],
],
[
    [1.5, 4.8, 8.2, 1.0],
    [3.8, 7.1, 1.0, 3.8],
    [6.1, 9.5, 2.8, 6.1],
    [8.4, 1.8, 5.1, 8.4],
],
[
    [1.0, 4.1, 7.4, 1.0],
    [3.1, 6.4, 9.8, 3.1],
    [5.4, 8.7, 2.1, 5.4],
    [7.7, 1.1, 4.4, 7.7],
],
[
    [2.0, 5.4, 8.7, 1.0],
    [4.4, 7.7, 1.0, 4.4],
    [6.7, 9.0, 2.4, 6.7],
    [8.0, 1.3, 4.7, 8.0],
],
[
    [3.3, 6.7, 10.0, 1.3],
    [4.7, 8.0, 1.3, 4.6],
    [7.0, 9.3, 2.6, 7.0],
    [9.3, 1.6, 5.0, 9.3],
],
[
    [1.6, 4.9, 8.3, 1.0],
    [3.9, 7.3, 1.0, 3.9],
    [6.2, 9.6, 2.9, 6.2],
    [8.6, 1.9, 5.2, 8.6],
],
[
    [1.0, 4.2, 7.5, 1.0],
    [3.2, 6.5, 9.9, 3.2],
    [5.5, 8.9, 2.2, 5.5],
    [7.8, 1.2, 4.5, 7.8],
],
[
    [2.2, 5.5, 8.8, 1.2],
    [4.5, 7.8, 1.1, 4.5],
    [6.8, 9.1, 2.5, 6.8],
    [9.1, 1.5, 4.8, 9.1],
],
[
    [3.5, 6.8, 1.0, 3.4],
    [5.8, 9.1, 2.4, 5.8],
    [8.1, 1.0, 4.8, 8.1],
    [1.0, 3.7, 7.1, 1.0],
],
[
    [28.5, 1.0, 1.0, 1.0],
    [24.3, 1.0, 100.0, 19.7],
    [33.7, 1.0, 86.2, 78.8],
    [99.1, 1.0, 1.0, 64.4],
],
[
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 64.2],
    [1.0, 1.0, 1.0, 88.3],
    [1.0, 1.0, 5.9, 9.4],
],
[
    [1.0, 86.6, 1.0, 1.0],
    [1.0, 9.1, 26.9, 1.0],
    [46.3, 5.0, 39.0, 1.0],
    [11.7, 100.0, 93.3, 1.0],
],
[
    [1.0, 53.2, 1.0, 21.5],
    [16.2, 9.5, 34.0, 47.5],
    [96.8, 77.7, 6.4, 5.3],
    [59.8, 29.3, 1.0, 1.0],
],
[
    [13.6, 1.0, 1.0, 1.0],
    [48.2, 4.3, 1.0, 35.8],
    [77.6, 100.0, 1.0, 100.0],
    [17.4, 86.7, 1.0, 4.1],
],
[
    [1.5, 1.0, 25.8, 1.0],
    [75.5, 100.0, 62.8, 16.4],
    [6.6, 40.2, 30.6, 100.0],
    [100.0, 4.0, 1.0, 1.0],
],
[
    [1.0, 1.0, 1.0, 1.0],
    [98.8, 1.0, 1.0, 1.0],
    [12.3, 12.2, 1.0, 1.0],
    [1.0, 9.4, 1.0, 1.0],
],
[
    [40.7, 1.0, 1.0, 1.0],
    [93.1, 44.6, 15.9, 17.3],
    [75.0, 3.3, 39.5, 54.4],
    [78.8, 88.1, 1.0, 7.3],
],
[
    [1.0, 1.0, 12.7, 55.9],
    [7.9, 52.6, 8.0, 35.4],
    [26.0, 23.5, 7.7, 39.3],
    [7.5, 3.2, 3.4, 61.5],
],
[
    [7.4, 41.7, 100.0, 2.7],
    [8.3, 25.7, 7.1, 9.1],
    [59.6, 66.9, 100.0, 41.5],
    [43.1, 2.7, 2.2, 59.8],
]
]


num_examples = int(len(matrices) / 2)  # Number of 4x4 matrix multiplication examples

# Define the alphatensor template for Bril code, created with assistance from
#   ChatGPT4 August 3 2023 version and healthy partial print debugging with
#   manual edits and a few optimizations
# parsing to translate and common subexpression elimination would be helpful here
template_str = '''
@main {
  {% for i in range(num_examples) %}

  {% for row in range(4) %}
  {% for col in range(4) %}
  a{{row+1}}{{col+1}}_{{i}}: float = const {{matrices[2*i][row][col]}};
  b{{row+1}}{{col+1}}_{{i}}: float = const {{matrices[2*i+1][row][col]}};
  {% endfor %}
  {% endfor %}
  n1: float = const -1;

  h1_part1_{{i}}: float = fadd a11_{{i}} a31_{{i}};
  h1_part2_{{i}}: float = fadd b11_{{i}} b31_{{i}};
  h1_{{i}}: float = fmul h1_part1_{{i}} h1_part2_{{i}};
#  print h1_{{i}};

  h2_part1_{{i}}: float = fsub a11_{{i}} a13_{{i}};
  h2_part2_{{i}}: float = fadd h2_part1_{{i}} a31_{{i}};
  h2_part3_{{i}}: float = fsub b11_{{i}} b13_{{i}};
  h2_part4_{{i}}: float = fadd h2_part3_{{i}} b31_{{i}};
  h2_{{i}}: float = fmul h2_part2_{{i}} h2_part4_{{i}};
#  print h2_{{i}};

  h3_part1_{{i}}: float = fsub b11_{{i}} b13_{{i}};
  h3_part2_{{i}}: float = fadd h3_part1_{{i}} b31_{{i}};
  h3_part3_{{i}}: float = fsub h3_part2_{{i}} b33_{{i}};
  h3_part4_{{i}}: float = fmul a13_{{i}} h3_part3_{{i}};
  h3_{{i}}: float = fmul h3_part4_{{i}} n1;
#  print h3_{{i}};

  h4_part1_{{i}}: float = fmul a33_{{i}} n1;
  h4_part2_{{i}}: float = fmul b33_{{i}} n1;
  h4_{{i}}: float = fmul h4_part1_{{i}} h4_part2_{{i}};
#  print h4_{{i}};

  h5_part1_{{i}}: float = fmul a31_{{i}} n1;
  h5_part2_{{i}}: float = fmul b13_{{i}} n1;
  h5_{{i}}: float = fmul h5_part1_{{i}} h5_part2_{{i}};
#  print h5_{{i}};

  h6_part1_{{i}}: float = fsub a11_{{i}} a13_{{i}};
  h6_part2_{{i}}: float = fadd h6_part1_{{i}} a31_{{i}};
  h6_part3_{{i}}: float = fsub h6_part2_{{i}} a33_{{i}};
  h6_part4_{{i}}: float = fmul b31_{{i}} n1;
  h6_{{i}}: float = fmul h6_part3_{{i}} h6_part4_{{i}};
#  print h6_{{i}};

  h7_part1_{{i}}: float = fsub a22_{{i}} a21_{{i}};
  h7_part2_{{i}}: float = fsub h7_part1_{{i}} a23_{{i}};
  h7_part3_{{i}}: float = fsub h7_part2_{{i}} a24_{{i}};
  h7_part4_{{i}}: float = fsub b22_{{i}} b21_{{i}};
  h7_part5_{{i}}: float = fsub h7_part4_{{i}} b23_{{i}};
  h7_part6_{{i}}: float = fsub h7_part5_{{i}} b24_{{i}};
  h7_{{i}}: float = fmul h7_part3_{{i}} h7_part6_{{i}};
#  print h7_{{i}};

  h8_part1_{{i}}: float = fsub a22_{{i}} a21_{{i}};
  h8_part2_{{i}}: float = fsub h8_part1_{{i}} a23_{{i}};
  h8_part3_{{i}}: float = fsub h8_part2_{{i}} a24_{{i}};
  h8_part4_{{i}}: float = fsub h8_part3_{{i}} a41_{{i}};
  h8_part5_{{i}}: float = fadd h8_part4_{{i}} a42_{{i}};
  h8_part6_{{i}}: float = fsub b22_{{i}} b21_{{i}};
  h8_part7_{{i}}: float = fsub h8_part6_{{i}} b23_{{i}};
  h8_part8_{{i}}: float = fsub h8_part7_{{i}} b24_{{i}};
  h8_part9_{{i}}: float = fsub h8_part8_{{i}} b41_{{i}};
  h8_part10_{{i}}: float = fadd h8_part9_{{i}} b42_{{i}};
  h8_{{i}}: float = fmul h8_part5_{{i}} h8_part10_{{i}};
#  print h8_{{i}};

  h9_part1_{{i}}: float = fsub a11_{{i}} a13_{{i}};
  h9_part2_{{i}}: float = fsub b11_{{i}} b13_{{i}};
  h9_{{i}}: float = fmul h9_part1_{{i}} h9_part2_{{i}};
#  print h9_{{i}};

  h10_part1_{{i}}: float = fsub a22_{{i}} a21_{{i}};
  h10_part2_{{i}}: float = fsub h10_part1_{{i}} a41_{{i}};
  h10_part3_{{i}}: float = fadd h10_part2_{{i}} a42_{{i}};
  h10_part4_{{i}}: float = fsub b22_{{i}} b21_{{i}};
  h10_part5_{{i}}: float = fsub h10_part4_{{i}} b41_{{i}};
  h10_part6_{{i}}: float = fadd h10_part5_{{i}} b42_{{i}};
  h10_{{i}}: float = fmul h10_part3_{{i}} h10_part6_{{i}};
#  print h10_{{i}};

  h11_part1_{{i}}: float = fsub a41_{{i}} a42_{{i}};
  h11_part2_{{i}}: float = fadd b23_{{i}} b24_{{i}};
  h11_part3_{{i}}: float = fmul n1 h11_part2_{{i}};
  h11_{{i}}: float = fmul h11_part1_{{i}} h11_part3_{{i}};
#  print h11_{{i}};

  h12_part1_{{i}}: float = fsub a22_{{i}} a21_{{i}};
  h12_part2_{{i}}: float = fsub h12_part1_{{i}} a23_{{i}};
  h12_part3_{{i}}: float = fsub h12_part2_{{i}} a24_{{i}};
  h12_part4_{{i}}: float = fsub h12_part3_{{i}} a41_{{i}};
  h12_part5_{{i}}: float = fadd h12_part4_{{i}} a42_{{i}};
  h12_part6_{{i}}: float = fsub h12_part5_{{i}} a43_{{i}};
  h12_part7_{{i}}: float = fsub h12_part6_{{i}} a44_{{i}};
  h12_part8_{{i}}: float = fsub b41_{{i}} b42_{{i}};
  h12_{{i}}: float = fmul h12_part7_{{i}} h12_part8_{{i}};
#  print h12_{{i}};

  h13_part1_{{i}}: float = fadd a23_{{i}} a24_{{i}};
  h13_part2_{{i}}: float = fmul n1 h13_part1_{{i}};
  h13_part3_{{i}}: float = fmul n1 b21_{{i}};
  h13_part4_{{i}}: float = fadd h13_part3_{{i}} b22_{{i}};
  h13_part5_{{i}}: float = fsub h13_part4_{{i}} b23_{{i}};
  h13_part6_{{i}}: float = fsub h13_part5_{{i}} b24_{{i}};
  h13_part7_{{i}}: float = fsub h13_part6_{{i}} b41_{{i}};
  h13_part8_{{i}}: float = fadd h13_part7_{{i}} b42_{{i}};
  h13_part9_{{i}}: float = fsub h13_part8_{{i}} b43_{{i}};
  h13_part10_{{i}}: float = fsub h13_part9_{{i}} b44_{{i}};
  h13_{{i}}: float = fmul h13_part2_{{i}} h13_part10_{{i}};
#  print h13_{{i}};

  h14_part1_{{i}}: float = fsub a11_{{i}} a12_{{i}};
  h14_part2_{{i}}: float = fadd h14_part1_{{i}} a21_{{i}};
  h14_part3_{{i}}: float = fsub h14_part2_{{i}} a22_{{i}};
  h14_part4_{{i}}: float = fmul n1 b12_{{i}};
  h14_part5_{{i}}: float = fsub h14_part4_{{i}} b14_{{i}};
  h14_{{i}}: float = fmul h14_part3_{{i}} h14_part5_{{i}};
#  print h14_{{i}};

  h15_part1_{{i}}: float = fadd a14_{{i}} a12_{{i}};
  h15_{{i}}: float = fmul h15_part1_{{i}} b21_{{i}};
#  print h15_{{i}};

  h16_part1_{{i}}: float = fadd a12_{{i}} a14_{{i}};
  h16_part2_{{i}}: float = fsub h16_part1_{{i}} a21_{{i}};
  h16_part3_{{i}}: float = fadd h16_part2_{{i}} a22_{{i}};
  h16_part4_{{i}}: float = fadd h16_part3_{{i}} a23_{{i}};
  h16_part5_{{i}}: float = fadd h16_part4_{{i}} a24_{{i}};
  h16_part6_{{i}}: float = fadd b12_{{i}} b14_{{i}};
  h16_part7_{{i}}: float = fsub h16_part6_{{i}} b21_{{i}};
  h16_part8_{{i}}: float = fadd h16_part7_{{i}} b22_{{i}};
  h16_part9_{{i}}: float = fadd h16_part8_{{i}} b23_{{i}};
  h16_part10_{{i}}: float = fadd h16_part9_{{i}} b24_{{i}};
  h16_{{i}}: float = fmul h16_part5_{{i}} h16_part10_{{i}};
#  print h16_{{i}};

  h17_part1_{{i}}: float = fadd a12_{{i}} a14_{{i}};
  h17_part2_{{i}}: float = fsub h17_part1_{{i}} a21_{{i}};
  h17_part3_{{i}}: float = fadd h17_part2_{{i}} a22_{{i}};
  h17_part4_{{i}}: float = fadd h17_part3_{{i}} a23_{{i}};
  h17_part5_{{i}}: float = fadd h17_part4_{{i}} a24_{{i}};
  h17_part6_{{i}}: float = fadd h17_part5_{{i}} a32_{{i}};
  h17_part7_{{i}}: float = fadd h17_part6_{{i}} a41_{{i}};
  h17_part8_{{i}}: float = fsub h17_part7_{{i}} a42_{{i}};
  h17_part9_{{i}}: float = fadd b12_{{i}} b14_{{i}};
  h17_part10_{{i}}: float = fsub h17_part9_{{i}} b21_{{i}};
  h17_part11_{{i}}: float = fadd h17_part10_{{i}} b22_{{i}};
  h17_part12_{{i}}: float = fadd h17_part11_{{i}} b23_{{i}};
  h17_part13_{{i}}: float = fadd h17_part12_{{i}} b24_{{i}};
  h17_part14_{{i}}: float = fadd h17_part13_{{i}} b32_{{i}};
  h17_part15_{{i}}: float = fadd h17_part14_{{i}} b41_{{i}};
  h17_part16_{{i}}: float = fsub h17_part15_{{i}} b42_{{i}};
  h17_{{i}}: float = fmul h17_part8_{{i}} h17_part16_{{i}};
#  print h17_{{i}};

  h18_part1_{{i}}: float = fsub a12_{{i}} a21_{{i}};
  h18_part2_{{i}}: float = fadd h18_part1_{{i}} a22_{{i}};
  h18_part3_{{i}}: float = fadd h18_part2_{{i}} a32_{{i}};
  h18_part4_{{i}}: float = fadd h18_part3_{{i}} a41_{{i}};
  h18_part5_{{i}}: float = fsub h18_part4_{{i}} a42_{{i}};
  h18_part6_{{i}}: float = fsub b12_{{i}} b21_{{i}};
  h18_part7_{{i}}: float = fadd h18_part6_{{i}} b22_{{i}};
  h18_part8_{{i}}: float = fadd h18_part7_{{i}} b32_{{i}};
  h18_part9_{{i}}: float = fadd h18_part8_{{i}} b41_{{i}};
  h18_part10_{{i}}: float = fsub h18_part9_{{i}} b42_{{i}};
  h18_{{i}}: float = fmul h18_part5_{{i}} h18_part10_{{i}};
#  print h18_{{i}};

  h19_part1_{{i}}: float = fadd a14_{{i}} a23_{{i}};
  h19_part2_{{i}}: float = fadd h19_part1_{{i}} a24_{{i}};
  h19_part3_{{i}}: float = fadd b12_{{i}} b14_{{i}};
  h19_part4_{{i}}: float = fsub h19_part3_{{i}} b21_{{i}};
  h19_part5_{{i}}: float = fadd h19_part4_{{i}} b22_{{i}};
  h19_part6_{{i}}: float = fadd h19_part5_{{i}} b23_{{i}};
  h19_part7_{{i}}: float = fadd h19_part6_{{i}} b24_{{i}};
  h19_part8_{{i}}: float = fadd h19_part7_{{i}} b32_{{i}};
  h19_part9_{{i}}: float = fadd h19_part8_{{i}} b34_{{i}};
  h19_part10_{{i}}: float = fadd h19_part9_{{i}} b41_{{i}};
  h19_part11_{{i}}: float = fsub h19_part10_{{i}} b42_{{i}};
  h19_part12_{{i}}: float = fsub h19_part11_{{i}} b43_{{i}};
  h19_part13_{{i}}: float = fsub h19_part12_{{i}} b44_{{i}};
  h19_{{i}}: float = fmul h19_part2_{{i}} h19_part13_{{i}};
#  print h19_{{i}};

  h20_part1_{{i}}: float = fadd a12_{{i}} a14_{{i}};
  h20_part2_{{i}}: float = fsub h20_part1_{{i}} a21_{{i}};
  h20_part3_{{i}}: float = fadd h20_part2_{{i}} a22_{{i}};
  h20_part4_{{i}}: float = fadd h20_part3_{{i}} a23_{{i}};
  h20_part5_{{i}}: float = fadd h20_part4_{{i}} a24_{{i}};
  h20_part6_{{i}}: float = fadd h20_part5_{{i}} a32_{{i}};
  h20_part7_{{i}}: float = fadd h20_part6_{{i}} a34_{{i}};
  h20_part8_{{i}}: float = fadd h20_part7_{{i}} a41_{{i}};
  h20_part9_{{i}}: float = fsub h20_part8_{{i}} a42_{{i}};
  h20_part10_{{i}}: float = fsub h20_part9_{{i}} a43_{{i}};
  h20_part11_{{i}}: float = fsub h20_part10_{{i}} a44_{{i}};
  h20_part12_{{i}}: float = fadd b32_{{i}} b41_{{i}};
  h20_part13_{{i}}: float = fsub h20_part12_{{i}} b42_{{i}};
  h20_{{i}}: float = fmul h20_part11_{{i}} h20_part13_{{i}};
#  print h20_{{i}};

  h21_part1_{{i}}: float = fadd a32_{{i}} a41_{{i}};
  h21_part2_{{i}}: float = fsub h21_part1_{{i}} a42_{{i}};
  h21_part3_{{i}}: float = fadd b14_{{i}} b23_{{i}};
  h21_part4_{{i}}: float = fadd h21_part3_{{i}} b24_{{i}};
  h21_{{i}}: float = fmul h21_part2_{{i}} h21_part4_{{i}};
#  print h21_{{i}};

  h22_part1_{{i}}: float = fadd a12_{{i}} a14_{{i}};
  h22_part2_{{i}}: float = fadd h22_part1_{{i}} a22_{{i}};
  h22_part3_{{i}}: float = fadd h22_part2_{{i}} a24_{{i}};
  h22_part4_{{i}}: float = fadd b12_{{i}} b14_{{i}};
  h22_part5_{{i}}: float = fadd h22_part4_{{i}} b22_{{i}};
  h22_part6_{{i}}: float = fadd h22_part5_{{i}} b24_{{i}};
  h22_{{i}}: float = fmul h22_part3_{{i}} h22_part6_{{i}};
#  print h22_{{i}};

  h23_part1_{{i}}: float = fadd a12_{{i}} a14_{{i}};
  h23_part2_{{i}}: float = fadd h23_part1_{{i}} a22_{{i}};
  h23_part3_{{i}}: float = fadd h23_part2_{{i}} a24_{{i}};
  h23_part4_{{i}}: float = fadd h23_part3_{{i}} a32_{{i}};
  h23_part5_{{i}}: float = fsub h23_part4_{{i}} a42_{{i}};
  h23_part6_{{i}}: float = fadd b12_{{i}} b14_{{i}};
  h23_part7_{{i}}: float = fadd h23_part6_{{i}} b22_{{i}};
  h23_part8_{{i}}: float = fadd h23_part7_{{i}} b24_{{i}};
  h23_part9_{{i}}: float = fadd h23_part8_{{i}} b32_{{i}};
  h23_part10_{{i}}: float = fsub h23_part9_{{i}} b42_{{i}};
  h23_{{i}}: float = fmul h23_part5_{{i}} h23_part10_{{i}};
#  print h23_{{i}};
  
  h24_part1_{{i}}: float = fadd a14_{{i}} a24_{{i}};
  h24_part2_{{i}}: float = fadd b12_{{i}} b14_{{i}};
  h24_part3_{{i}}: float = fadd h24_part2_{{i}} b22_{{i}};
  h24_part4_{{i}}: float = fadd h24_part3_{{i}} b24_{{i}};
  h24_part5_{{i}}: float = fadd h24_part4_{{i}} b32_{{i}};
  h24_part6_{{i}}: float = fadd h24_part5_{{i}} b34_{{i}};
  h24_part7_{{i}}: float = fsub h24_part6_{{i}} b42_{{i}};
  h24_part8_{{i}}: float = fsub h24_part7_{{i}} b44_{{i}};
  h24_{{i}}: float = fmul h24_part1_{{i}} h24_part8_{{i}};
#  print h24_{{i}};
  
  h25_part1_{{i}}: float = fadd a12_{{i}} a14_{{i}};
  h25_part2_{{i}}: float = fadd h25_part1_{{i}} a22_{{i}};
  h25_part3_{{i}}: float = fadd h25_part2_{{i}} a24_{{i}};
  h25_part4_{{i}}: float = fadd h25_part3_{{i}} a32_{{i}};
  h25_part5_{{i}}: float = fadd h25_part4_{{i}} a34_{{i}};
  h25_part6_{{i}}: float = fsub h25_part5_{{i}} a42_{{i}};
  h25_part7_{{i}}: float = fsub h25_part6_{{i}} a44_{{i}};
  h25_part8_{{i}}: float = fsub b32_{{i}} b42_{{i}};
  h25_{{i}}: float = fmul h25_part7_{{i}} h25_part8_{{i}};
#  print h25_{{i}};
  
  h26_part1_{{i}}: float = fsub a32_{{i}} a42_{{i}};
  h26_part2_{{i}}: float = fadd b14_{{i}} b24_{{i}};
  h26_{{i}}: float = fmul h26_part1_{{i}} h26_part2_{{i}};
#  print h26_{{i}};
  
  h27_part1_{{i}}: float = fsub a34_{{i}} a44_{{i}};
  h27_part2_{{i}}: float = fsub b34_{{i}} b44_{{i}};
  h27_{{i}}: float = fmul h27_part1_{{i}} h27_part2_{{i}};
#  print h27_{{i}};
  
  h28_part1_{{i}}: float = fsub a34_{{i}} a43_{{i}};
  h28_part2_{{i}}: float = fsub h28_part1_{{i}} a44_{{i}};
  h28_part3_{{i}}: float = fsub b34_{{i}} b43_{{i}};
  h28_part4_{{i}}: float = fsub h28_part3_{{i}} b44_{{i}};
  h28_{{i}}: float = fmul h28_part2_{{i}} h28_part4_{{i}};
#  print h28_{{i}};

  h29_part1_{{i}}: float = fadd a14_{{i}} a34_{{i}};
  h29_part2_{{i}}: float = fmul n1 b43_{{i}};
  h29_{{i}}: float = fmul h29_part1_{{i}} h29_part2_{{i}};
#  print h29_{{i}};

  h30_part1_{{i}}: float = fadd a13_{{i}} a14_{{i}};
  h30_part2_{{i}}: float = fadd h30_part1_{{i}} a23_{{i}};
  h30_part3_{{i}}: float = fadd h30_part2_{{i}} a24_{{i}};
  h30_part4_{{i}}: float = fadd h30_part3_{{i}} a33_{{i}};
  h30_part5_{{i}}: float = fadd h30_part4_{{i}} a34_{{i}};
  h30_part6_{{i}}: float = fsub h30_part5_{{i}} a43_{{i}};
  h30_part7_{{i}}: float = fsub h30_part6_{{i}} a44_{{i}};
  h30_part8_{{i}}: float = fadd b14_{{i}} b34_{{i}};
  h30_{{i}}: float = fmul h30_part7_{{i}} h30_part8_{{i}};
#  print h30_{{i}};

  h31_part1_{{i}}: float = fsub a11_{{i}} a12_{{i}};
  h31_part2_{{i}}: float = fsub h31_part1_{{i}} a13_{{i}};
  h31_part3_{{i}}: float = fsub h31_part2_{{i}} a14_{{i}};
  h31_part4_{{i}}: float = fadd h31_part3_{{i}} a21_{{i}};
  h31_part5_{{i}}: float = fsub h31_part4_{{i}} a22_{{i}};
  h31_part6_{{i}}: float = fsub h31_part5_{{i}} a23_{{i}};
  h31_part7_{{i}}: float = fsub h31_part6_{{i}} a24_{{i}};
  h31_part8_{{i}}: float = fadd h31_part7_{{i}} a31_{{i}};
  h31_part9_{{i}}: float = fsub h31_part8_{{i}} a32_{{i}};
  h31_part10_{{i}}: float = fsub h31_part9_{{i}} a33_{{i}};
  h31_part11_{{i}}: float = fsub h31_part10_{{i}} a34_{{i}};
  h31_part12_{{i}}: float = fsub h31_part11_{{i}} a41_{{i}};
  h31_part13_{{i}}: float = fadd h31_part12_{{i}} a42_{{i}};
  h31_part14_{{i}}: float = fadd h31_part13_{{i}} a43_{{i}};
  h31_part15_{{i}}: float = fadd h31_part14_{{i}} a44_{{i}};
  h31_{{i}}: float = fmul h31_part15_{{i}} b14_{{i}};
#  print h31_{{i}};

  h32_part1_{{i}}: float = fadd b13_{{i}} b14_{{i}};
  h32_part2_{{i}}: float = fadd h32_part1_{{i}} b23_{{i}};
  h32_part3_{{i}}: float = fadd h32_part2_{{i}} b24_{{i}};
  h32_part4_{{i}}: float = fadd h32_part3_{{i}} b33_{{i}};
  h32_part5_{{i}}: float = fadd h32_part4_{{i}} b34_{{i}};
  h32_part6_{{i}}: float = fsub h32_part5_{{i}} b43_{{i}};
  h32_part7_{{i}}: float = fsub h32_part6_{{i}} b44_{{i}};
  h32_{{i}}: float = fmul n1 a43_{{i}};
  h32_{{i}}: float = fmul h32_{{i}} h32_part7_{{i}};
#  print h32_{{i}};

  h33_part1_{{i}}: float = fmul n1 b21_{{i}};
  h33_part2_{{i}}: float = fadd b41_{{i}} h33_part1_{{i}};
  h33_{{i}}: float = fmul a14_{{i}} h33_part2_{{i}};
#  print h33_{{i}};

  h34_part1_{{i}}: float = fsub a14_{{i}} a32_{{i}};
  h34_part2_{{i}}: float = fsub h33_part2_{{i}} b43_{{i}};
  h34_{{i}}: float = fmul h34_part1_{{i}} h34_part2_{{i}};
#  print h34_{{i}};

  h35_part1_{{i}}: float = fadd a13_{{i}} a14_{{i}};
  h35_part2_{{i}}: float = fadd h35_part1_{{i}} a23_{{i}};
  h35_part3_{{i}}: float = fadd h35_part2_{{i}} a24_{{i}};
  h35_part4_{{i}}: float = fsub h35_part3_{{i}} a31_{{i}};
  h35_part5_{{i}}: float = fadd h35_part4_{{i}} a32_{{i}};
  h35_part6_{{i}}: float = fadd h35_part5_{{i}} a33_{{i}};
  h35_part7_{{i}}: float = fadd h35_part6_{{i}} a34_{{i}};
  h35_part8_{{i}}: float = fadd h35_part7_{{i}} a41_{{i}};
  h35_part9_{{i}}: float = fsub h35_part8_{{i}} a42_{{i}};
  h35_part10_{{i}}: float = fsub h35_part9_{{i}} a43_{{i}};
  h35_part11_{{i}}: float = fsub h35_part10_{{i}} a44_{{i}};
  h35_part12_{{i}}: float = fsub b14_{{i}} b32_{{i}};
  h35_{{i}}: float = fmul h35_part11_{{i}} h35_part12_{{i}};
#  print h35_{{i}};

  h36_part1_{{i}}: float = fsub a32_{{i}} a31_{{i}};
  h36_part3_{{i}}: float = fadd h36_part1_{{i}} a33_{{i}};
  h36_part4_{{i}}: float = fadd h36_part3_{{i}} a34_{{i}};
  h36_part5_{{i}}: float = fadd h36_part4_{{i}} a41_{{i}};
  h36_part6_{{i}}: float = fsub h36_part5_{{i}} a42_{{i}};
  h36_part7_{{i}}: float = fsub h36_part6_{{i}} a43_{{i}};
  h36_part8_{{i}}: float = fsub h36_part7_{{i}} a44_{{i}};
  h36_{{i}}: float = fmul h36_part8_{{i}} b32_{{i}};
#  print h36_{{i}};

  h37_part1_{{i}}: float = fadd a12_{{i}} a32_{{i}};
  h37_{{i}}: float = fmul b23_{{i}} h37_part1_{{i}};
#  p37: int = const 37;
#  print p37;
#  print h37_{{i}};
  
  h38_part1_{{i}}: float = fadd a32_{{i}} a34_{{i}};
  h38_part2_{{i}}: float = fsub b41_{{i}} b43_{{i}};
  h38_{{i}}: float = fmul h38_part1_{{i}} h38_part2_{{i}};
#  print h38_{{i}};
  
  h39_part1_{{i}}: float = fadd a13_{{i}} a14_{{i}};
  h39_part2_{{i}}: float = fadd h39_part1_{{i}} a23_{{i}};
  h39_part3_{{i}}: float = fadd h39_part2_{{i}} a24_{{i}};
  h39_part4_{{i}}: float = fmul n1 h39_part3_{{i}};
  h39_part5_{{i}}: float = fadd b32_{{i}} b34_{{i}};
  h39_{{i}}: float = fmul h39_part4_{{i}} h39_part5_{{i}};
#  p39: int = const 39;
#  print p39;
#  print h39_{{i}};
  
  h40_part2_{{i}}: float = fsub b23_{{i}} b21_{{i}};
  h40_part3_{{i}}: float = fadd h40_part2_{{i}} b41_{{i}};
  h40_part4_{{i}}: float = fsub h40_part3_{{i}} b43_{{i}};
  h40_{{i}}: float = fmul a32_{{i}} h40_part4_{{i}};
#  print h40_{{i}};
  
  h41_part1_{{i}}: float = fmul a21_{{i}} n1;
  h41_part2_{{i}}: float = fsub b11_{{i}} b12_{{i}};
  h41_part3_{{i}}: float = fadd h41_part2_{{i}} b21_{{i}};
  h41_part4_{{i}}: float = fsub h41_part3_{{i}} b22_{{i}};
  h41_{{i}}: float = fmul h41_part1_{{i}} h41_part4_{{i}};
#  print h41_{{i}};
  
  h42_part1_{{i}}: float = fmul a21_{{i}} n1;
  h42_part2_{{i}}: float = fadd h42_part1_{{i}} a41_{{i}};
  h42_part3_{{i}}: float = fsub b11_{{i}} b12_{{i}};
  h42_part4_{{i}}: float = fsub h42_part3_{{i}} b13_{{i}};
  h42_part5_{{i}}: float = fsub h42_part4_{{i}} b14_{{i}};
  h42_part6_{{i}}: float = fadd h42_part5_{{i}} b21_{{i}};
  h42_part7_{{i}}: float = fsub h42_part6_{{i}} b22_{{i}};
  h42_part8_{{i}}: float = fsub h42_part7_{{i}} b23_{{i}};
  h42_part9_{{i}}: float = fsub h42_part8_{{i}} b24_{{i}};
  h42_part10_{{i}}: float = fadd h42_part9_{{i}} b31_{{i}};
  h42_part11_{{i}}: float = fsub h42_part10_{{i}} b32_{{i}};
  h42_part12_{{i}}: float = fsub h42_part11_{{i}} b33_{{i}};
  h42_part13_{{i}}: float = fsub h42_part12_{{i}} b34_{{i}};
  h42_part14_{{i}}: float = fsub h42_part13_{{i}} b41_{{i}};
  h42_part15_{{i}}: float = fadd h42_part14_{{i}} b42_{{i}};
  h42_part16_{{i}}: float = fadd h42_part15_{{i}} b43_{{i}};
  h42_part17_{{i}}: float = fadd h42_part16_{{i}} b44_{{i}};
  h42_{{i}}: float = fmul h42_part2_{{i}} h42_part17_{{i}};
#  print h42_{{i}};
  
  h43_part1_{{i}}: float = fmul a21_{{i}} n1;
  h43_part2_{{i}}: float = fadd h43_part1_{{i}} a41_{{i}};
  h43_part3_{{i}}: float = fsub h43_part2_{{i}} a43_{{i}};
  h43_part4_{{i}}: float = fadd b13_{{i}} b14_{{i}};
  h43_part5_{{i}}: float = fadd h43_part4_{{i}} b23_{{i}};
  h43_part6_{{i}}: float = fadd h43_part5_{{i}} b24_{{i}};
  h43_part7_{{i}}: float = fsub h43_part6_{{i}} b31_{{i}};
  h43_part8_{{i}}: float = fadd h43_part7_{{i}} b32_{{i}};
  h43_part9_{{i}}: float = fadd h43_part8_{{i}} b33_{{i}};
  h43_part10_{{i}}: float = fadd h43_part9_{{i}} b34_{{i}};
  h43_part11_{{i}}: float = fadd h43_part10_{{i}} b41_{{i}};
  h43_part12_{{i}}: float = fsub h43_part11_{{i}} b42_{{i}};
  h43_part13_{{i}}: float = fsub h43_part12_{{i}} b43_{{i}};
  h43_part14_{{i}}: float = fsub h43_part13_{{i}} b44_{{i}};
  h43_{{i}}: float = fmul h43_part3_{{i}} h43_part14_{{i}};
#  print h43_{{i}};
  
  h44_part1_{{i}}: float = fadd a12_{{i}} a22_{{i}};
  h44_part2_{{i}}: float = fadd h44_part1_{{i}} a32_{{i}};
  h44_part3_{{i}}: float = fsub h44_part2_{{i}} a42_{{i}};
  h44_part4_{{i}}: float = fadd b12_{{i}} b22_{{i}};
  h44_part5_{{i}}: float = fadd h44_part4_{{i}} b32_{{i}};
  h44_part6_{{i}}: float = fsub h44_part5_{{i}} b42_{{i}};
  h44_{{i}}: float = fmul h44_part3_{{i}} h44_part6_{{i}};
#  print h44_{{i}};
  
  h45_part1_{{i}}: float = fmul a21_{{i}} n1;
  h45_part2_{{i}}: float = fadd h45_part1_{{i}} a23_{{i}};
  h45_part3_{{i}}: float = fadd h45_part2_{{i}} a41_{{i}};
  h45_part4_{{i}}: float = fsub h45_part3_{{i}} a43_{{i}};
  h45_part5_{{i}}: float = fmul b31_{{i}} n1;
  h45_part6_{{i}}: float = fadd h45_part5_{{i}} b32_{{i}};
  h45_part7_{{i}}: float = fadd h45_part6_{{i}} b33_{{i}};
  h45_part8_{{i}}: float = fadd h45_part7_{{i}} b34_{{i}};
  h45_part9_{{i}}: float = fadd h45_part8_{{i}} b41_{{i}};
  h45_part10_{{i}}: float = fsub h45_part9_{{i}} b42_{{i}};
  h45_part11_{{i}}: float = fsub h45_part10_{{i}} b43_{{i}};
  h45_part12_{{i}}: float = fsub h45_part11_{{i}} b44_{{i}};
  h45_{{i}}: float = fmul h45_part4_{{i}} h45_part12_{{i}};
#  print h45_{{i}};
  
  h46_part1_{{i}}: float = fmul a31_{{i}} n1;
  h46_part2_{{i}}: float = fadd h46_part1_{{i}} a32_{{i}};
  h46_part3_{{i}}: float = fadd h46_part2_{{i}} a41_{{i}};
  h46_part4_{{i}}: float = fsub h46_part3_{{i}} a42_{{i}};
  h46_part5_{{i}}: float = fmul b12_{{i}} n1;
  h46_part6_{{i}}: float = fadd h46_part5_{{i}} b32_{{i}};
  h46_part7_{{i}}: float = fmul h46_part6_{{i}} n1;
  h46_{{i}}: float = fmul h46_part4_{{i}} h46_part7_{{i}};
#  print h46_{{i}};
  
  h47_part1_{{i}}: float = fsub a41_{{i}} a43_{{i}};
  h47_part2_{{i}}: float = fadd b13_{{i}} b14_{{i}};
  h47_part3_{{i}}: float = fadd h47_part2_{{i}} b23_{{i}};
  h47_part4_{{i}}: float = fadd h47_part3_{{i}} b24_{{i}};
  h47_part5_{{i}}: float = fmul h47_part4_{{i}} n1;
  h47_{{i}}: float = fmul h47_part1_{{i}} h47_part5_{{i}};
#  p47: int = const 47;
#  print p47;
#  print h47_{{i}};
  
  h48_part1_{{i}}: float = fadd a43_{{i}} a44_{{i}};
  h48_part2_{{i}}: float = fadd b43_{{i}} b44_{{i}};
  h48_{{i}}: float = fmul h48_part1_{{i}} h48_part2_{{i}};
#  print h48_{{i}};
  
  h49_part1_{{i}}: float = fmul a23_{{i}} n1;
  h49_part2_{{i}}: float = fmul b31_{{i}} n1;
  h49_part3_{{i}}: float = fadd h49_part2_{{i}} b32_{{i}};
  h49_part4_{{i}}: float = fadd h49_part3_{{i}} b41_{{i}};
  h49_part5_{{i}}: float = fsub h49_part4_{{i}} b42_{{i}};
  h49_{{i}}: float = fmul h49_part1_{{i}} h49_part5_{{i}};
#  print h49_{{i}};
  
  c11_part1_{{i}}: float = fsub h1_{{i}} h2_{{i}};
  c11_part2_{{i}}: float = fsub c11_part1_{{i}} h5_{{i}};
  c11_part3_{{i}}: float = fadd c11_part2_{{i}} h9_{{i}};
  c11_part4_{{i}}: float = fadd c11_part3_{{i}} h15_{{i}};
  c11_{{i}}: float = fadd c11_part4_{{i}} h33_{{i}};
#  p11: int = const 11;
#  print p11;
#  print c11_{{i}};

# literal transpose  
  c21_part1_{{i}}: float = fmul h15_{{i}} n1;
  c21_part2_{{i}}: float = fsub c21_part1_{{i}} h16_{{i}};
  c21_part3_{{i}}: float = fadd c21_part2_{{i}} h17_{{i}};
  c21_part4_{{i}}: float = fsub c21_part3_{{i}} h18_{{i}};
  c21_part5_{{i}}: float = fsub c21_part4_{{i}} h21_{{i}};
  c21_part6_{{i}}: float = fadd c21_part5_{{i}} h22_{{i}};
  c21_part7_{{i}}: float = fsub c21_part6_{{i}} h23_{{i}};
  c21_part8_{{i}}: float = fadd c21_part7_{{i}} h26_{{i}};
  c21_part9_{{i}}: float = fsub c21_part8_{{i}} h33_{{i}};
  c21_part10_{{i}}: float = fsub c21_part9_{{i}} h41_{{i}};
  c21_part11_{{i}}: float = fadd c21_part10_{{i}} h44_{{i}};
  c21_{{i}}: float = fadd c21_part11_{{i}} h49_{{i}};
#  print c21_{{i}};
  
  c31_part1_{{i}}: float = fadd h2_{{i}} h5_{{i}};
  c31_part2_{{i}}: float = fadd c31_part1_{{i}} h6_{{i}};
  c31_part3_{{i}}: float = fsub c31_part2_{{i}} h9_{{i}};
  c31_part4_{{i}}: float = fsub c31_part3_{{i}} h29_{{i}};
  c31_part5_{{i}}: float = fsub c31_part4_{{i}} h33_{{i}};
  c31_part6_{{i}}: float = fadd c31_part5_{{i}} h34_{{i}};
  c31_{{i}}: float = fadd c31_part6_{{i}} h38_{{i}};
#  print c31_{{i}};

  c41_part1_{{i}}: float = fmul n1 h16_{{i}};
  c41_{{i}}: float = fadd c41_part1_{{i}} h17_{{i}};
  c41_{{i}}: float = fsub c41_{{i}} h20_{{i}};
  c41_{{i}}: float = fsub c41_{{i}} h21_{{i}};
  c41_{{i}}: float = fadd c41_{{i}} h22_{{i}};
  c41_{{i}}: float = fsub c41_{{i}} h23_{{i}};
  c41_{{i}}: float = fadd c41_{{i}} h25_{{i}};
  c41_{{i}}: float = fadd c41_{{i}} h26_{{i}};
  c41_{{i}}: float = fsub c41_{{i}} h29_{{i}};
  c41_{{i}}: float = fsub c41_{{i}} h32_{{i}};
  c41_{{i}}: float = fsub c41_{{i}} h33_{{i}};
  c41_{{i}}: float = fadd c41_{{i}} h34_{{i}};
  c41_{{i}}: float = fadd c41_{{i}} h38_{{i}};
  c41_{{i}}: float = fsub c41_{{i}} h41_{{i}};
  c41_{{i}}: float = fadd c41_{{i}} h42_{{i}};
  c41_{{i}}: float = fadd c41_{{i}} h43_{{i}};
#  print c41_{{i}};

  c12_part1_{{i}}: float = fmul n1 h7_{{i}};
  c12_{{i}}: float = fadd c12_part1_{{i}} h8_{{i}};
  c12_{{i}}: float = fsub c12_{{i}} h10_{{i}};
  c12_{{i}}: float = fadd c12_{{i}} h11_{{i}};
  c12_{{i}}: float = fsub c12_{{i}} h14_{{i}};
  c12_{{i}}: float = fadd c12_{{i}} h15_{{i}};
  c12_{{i}}: float = fadd c12_{{i}} h16_{{i}};
  c12_{{i}}: float = fsub c12_{{i}} h17_{{i}};
  c12_{{i}}: float = fadd c12_{{i}} h18_{{i}};
  c12_{{i}}: float = fadd c12_{{i}} h21_{{i}};
  c12_{{i}}: float = fsub c12_{{i}} h31_{{i}};
  c12_{{i}}: float = fadd c12_{{i}} h33_{{i}};
  c12_{{i}}: float = fsub c12_{{i}} h35_{{i}};
  c12_{{i}}: float = fsub c12_{{i}} h36_{{i}};
#  p12: int = const 12;
#  print p12;
#  print c12_{{i}};

  c22_{{i}}: float = fsub h7_{{i}} h8_{{i}};
  c22_{{i}}: float = fadd c22_{{i}} h10_{{i}};
  c22_{{i}}: float = fsub c22_{{i}} h11_{{i}};
  c22_{{i}}: float = fsub c22_{{i}} h15_{{i}};
  c22_{{i}}: float = fsub c22_{{i}} h16_{{i}};
  c22_{{i}}: float = fadd c22_{{i}} h17_{{i}};
  c22_{{i}}: float = fsub c22_{{i}} h18_{{i}};
  c22_{{i}}: float = fsub c22_{{i}} h21_{{i}};
  c22_{{i}}: float = fadd c22_{{i}} h22_{{i}};
  c22_{{i}}: float = fsub c22_{{i}} h23_{{i}};
  c22_{{i}}: float = fadd c22_{{i}} h26_{{i}};
  c22_{{i}}: float = fsub c22_{{i}} h33_{{i}};
  c22_{{i}}: float = fadd c22_{{i}} h44_{{i}};
#  print c22_{{i}};

  c32_part1_{{i}}: float = fmul n1 h7_{{i}};
  c32_{{i}}: float = fadd c32_part1_{{i}} h8_{{i}};
  c32_{{i}}: float = fadd c32_{{i}} h11_{{i}};
  c32_{{i}}: float = fadd c32_{{i}} h12_{{i}};
  c32_{{i}}: float = fsub c32_{{i}} h16_{{i}};
  c32_{{i}}: float = fadd c32_{{i}} h17_{{i}};
  c32_{{i}}: float = fsub c32_{{i}} h20_{{i}};
  c32_{{i}}: float = fsub c32_{{i}} h21_{{i}};
  c32_{{i}}: float = fsub c32_{{i}} h29_{{i}};
  c32_{{i}}: float = fsub c32_{{i}} h33_{{i}};
  c32_{{i}}: float = fadd c32_{{i}} h34_{{i}};
  c32_{{i}}: float = fadd c32_{{i}} h36_{{i}};
  c32_{{i}}: float = fadd c32_{{i}} h38_{{i}};
  c32_{{i}}: float = fadd c32_{{i}} h46_{{i}};
#  print c32_{{i}};

  c42_part1_{{i}}: float = fmul n1 h7_{{i}};
  c42_{{i}}: float = fadd c42_part1_{{i}} h8_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h11_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h12_{{i}};
  c42_{{i}}: float = fsub c42_{{i}} h16_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h17_{{i}};
  c42_{{i}}: float = fsub c42_{{i}} h20_{{i}};
  c42_{{i}}: float = fsub c42_{{i}} h21_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h22_{{i}};
  c42_{{i}}: float = fsub c42_{{i}} h23_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h25_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h26_{{i}};
  c42_{{i}}: float = fsub c42_{{i}} h29_{{i}};
  c42_{{i}}: float = fsub c42_{{i}} h33_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h34_{{i}};
  c42_{{i}}: float = fadd c42_{{i}} h38_{{i}};
#  print c42_{{i}};

  c13_{{i}}: float = fsub h1_{{i}} h2_{{i}};
  c13_{{i}}: float = fadd c13_{{i}} h3_{{i}};
  c13_{{i}}: float = fsub c13_{{i}} h5_{{i}};
  c13_{{i}}: float = fadd c13_{{i}} h33_{{i}};
  c13_{{i}}: float = fsub c13_{{i}} h34_{{i}};
  c13_{{i}}: float = fadd c13_{{i}} h37_{{i}};
  c13_{{i}}: float = fsub c13_{{i}} h40_{{i}};
#  p13: int = const 13;
#  print p13;
#  print c13_{{i}};

  c23_{{i}}: float = fsub h17_{{i}} h18_{{i}};
  c23_{{i}}: float = fsub c23_{{i}} h19_{{i}};
  c23_{{i}}: float = fsub c23_{{i}} h21_{{i}};
  c23_{{i}}: float = fsub c23_{{i}} h23_{{i}};
  c23_{{i}}: float = fadd c23_{{i}} h24_{{i}};
  c23_{{i}}: float = fadd c23_{{i}} h26_{{i}};
  c23_{{i}}: float = fsub c23_{{i}} h33_{{i}};
  c23_{{i}}: float = fadd c23_{{i}} h34_{{i}};
  c23_{{i}}: float = fsub c23_{{i}} h37_{{i}};
  c23_{{i}}: float = fadd c23_{{i}} h40_{{i}};
  c23_{{i}}: float = fsub c23_{{i}} h43_{{i}};
  c23_{{i}}: float = fadd c23_{{i}} h44_{{i}};
  c23_{{i}}: float = fadd c23_{{i}} h45_{{i}};
  c23_{{i}}: float = fsub c23_{{i}} h47_{{i}};
  c23_{{i}}: float = fadd c23_{{i}} h49_{{i}};
#  print c23_{{i}};

  c33_{{i}}: float = fadd h4_{{i}} h5_{{i}};
  c33_{{i}}: float = fsub c33_{{i}} h29_{{i}};
  c33_{{i}}: float = fsub c33_{{i}} h33_{{i}};
  c33_{{i}}: float = fadd c33_{{i}} h34_{{i}};
  c33_{{i}}: float = fadd c33_{{i}} h40_{{i}};
#  print c33_{{i}};

  c43_part1_{{i}}: float = fmul n1 h21_{{i}};
  c43_{{i}}: float = fadd c43_part1_{{i}} h26_{{i}};
  c43_{{i}}: float = fsub c43_{{i}} h27_{{i}};
  c43_{{i}}: float = fadd c43_{{i}} h28_{{i}};
  c43_{{i}}: float = fsub c43_{{i}} h29_{{i}};
  c43_{{i}}: float = fsub c43_{{i}} h32_{{i}};
  c43_{{i}}: float = fsub c43_{{i}} h33_{{i}};
  c43_{{i}}: float = fadd c43_{{i}} h34_{{i}};
  c43_{{i}}: float = fadd c43_{{i}} h40_{{i}};
  c43_{{i}}: float = fsub c43_{{i}} h47_{{i}};
#  print c43_{{i}};

  c14_{{i}}: float = fsub h8_{{i}} h10_{{i}};
  c14_{{i}}: float = fadd c14_{{i}} h11_{{i}};
  c14_{{i}}: float = fsub c14_{{i}} h13_{{i}};
  c14_{{i}}: float = fadd c14_{{i}} h17_{{i}};
  c14_{{i}}: float = fsub c14_{{i}} h18_{{i}};
  c14_{{i}}: float = fsub c14_{{i}} h19_{{i}};
  c14_{{i}}: float = fsub c14_{{i}} h21_{{i}};
  c14_{{i}}: float = fadd c14_{{i}} h31_{{i}};
  c14_{{i}}: float = fsub c14_{{i}} h33_{{i}};
  c14_{{i}}: float = fadd c14_{{i}} h34_{{i}};
  c14_{{i}}: float = fadd c14_{{i}} h35_{{i}};
  c14_{{i}}: float = fadd c14_{{i}} h36_{{i}};
  c14_{{i}}: float = fsub c14_{{i}} h37_{{i}};
  c14_{{i}}: float = fsub c14_{{i}} h39_{{i}};
  c14_{{i}}: float = fadd c14_{{i}} h40_{{i}};
#  print c14_{{i}};

  c24_part1_{{i}}: float = fmul n1 h8_{{i}};
  c24_{{i}}: float = fadd c24_part1_{{i}} h10_{{i}};
  c24_{{i}}: float = fsub c24_{{i}} h11_{{i}};
  c24_{{i}}: float = fadd c24_{{i}} h13_{{i}};
  c24_{{i}}: float = fsub c24_{{i}} h17_{{i}};
  c24_{{i}}: float = fadd c24_{{i}} h18_{{i}};
  c24_{{i}}: float = fadd c24_{{i}} h19_{{i}};
  c24_{{i}}: float = fadd c24_{{i}} h21_{{i}};
  c24_{{i}}: float = fadd c24_{{i}} h23_{{i}};
  c24_{{i}}: float = fsub c24_{{i}} h24_{{i}};
  c24_{{i}}: float = fsub c24_{{i}} h26_{{i}};
  c24_{{i}}: float = fadd c24_{{i}} h33_{{i}};
  c24_{{i}}: float = fsub c24_{{i}} h34_{{i}};
  c24_{{i}}: float = fadd c24_{{i}} h37_{{i}};
  c24_{{i}}: float = fsub c24_{{i}} h40_{{i}};
  c24_{{i}}: float = fsub c24_{{i}} h44_{{i}};
#  print c24_{{i}};

  c34_{{i}}: float = fadd h11_{{i}} h21_{{i}};
  c34_{{i}}: float = fsub c34_{{i}} h28_{{i}};
  c34_{{i}}: float = fadd c34_{{i}} h29_{{i}};
  c34_{{i}}: float = fadd c34_{{i}} h30_{{i}};
  c34_{{i}}: float = fadd c34_{{i}} h33_{{i}};
  c34_{{i}}: float = fsub c34_{{i}} h34_{{i}};
  c34_{{i}}: float = fsub c34_{{i}} h35_{{i}};
  c34_{{i}}: float = fsub c34_{{i}} h36_{{i}};
  c34_{{i}}: float = fadd c34_{{i}} h39_{{i}};
  c34_{{i}}: float = fsub c34_{{i}} h40_{{i}};
  c34_{{i}}: float = fadd c34_{{i}} h48_{{i}};
#  print c34_{{i}};
  
  c44_{{i}}: float = fadd h11_{{i}} h21_{{i}};
  c44_{{i}}: float = fsub c44_{{i}} h26_{{i}};
  c44_{{i}}: float = fadd c44_{{i}} h27_{{i}};
  c44_{{i}}: float = fsub c44_{{i}} h28_{{i}};
  c44_{{i}}: float = fadd c44_{{i}} h29_{{i}};
  c44_{{i}}: float = fadd c44_{{i}} h33_{{i}};
  c44_{{i}}: float = fsub c44_{{i}} h34_{{i}};
  c44_{{i}}: float = fsub c44_{{i}} h40_{{i}};
  c44_{{i}}: float = fadd c44_{{i}} h48_{{i}};
#  print c44_{{i}};

  {% for row in range(4) %}
  {% for col in range(4) %}
  print c{{row+1}}{{col+1}}_{{i}};
  {% endfor %}
  {% endfor %}

  {% endfor %}
  ret;
}
'''


# Generate reference results using NumPy and write to matmulti4x4.ref
with open("matmulti4x4.ref", "w") as ref_file:
    for i in range(num_examples):
        #print(f"i: {i}")
        mat1 = np.array(matrices[2*i])
        mat2 = np.array(matrices[2*i + 1])
        result = np.matmul(mat1, mat2)
        for row in result:
            for element in row:
                ref_file.write(f"{element:.17f}\n")

# Create a Jinja2 template and render it
template = Template(template_str)
rendered_str = template.render(matrices=matrices, num_examples=num_examples)

# Write the rendered Bril code to matmulti4x4.bril
with open("matmulti4x4.bril", "w") as bril_file:
    bril_file.write(rendered_str)



