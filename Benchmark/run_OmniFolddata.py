import os
import numpy as np

os.system("rm LHCO_tmp.root")
os.system("./DelphesPythia8 delphes_card_CMS_R4.tcl pythia_zjets.cmnd LHCO_tmp.root")
os.system("root -b -x -q 'myprocess_Omni.C(\"LHCO_tmp.root\",\"test_Omni.txt\")'")
