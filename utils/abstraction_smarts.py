# Description: This file contains the abstraction smarts for the dataset curation process.
from utils.electronegs import MORE_ELECTRONEG # atoms more electronegative than C

# preprocessing to prevent some structures from being abstracted
# leaving atoms not abstracted - tagged with 10 during preprocessing
preprocess_smarts = [
    # 0. Preprocessing

    # 0-1. Rearrangement: Non-ring middle atom leaving [core]!@[leaving_atom]!@[core]
    # Middle leaving atom connected to two core atoms through non-ring bonds - change isotope label to 10
    '[200*,201*,202*;A;$([*](!@[100*])!@[100*]):1]>>[10*:1]', 
    '[200*,201*,202*;A;$([*]~[10*;A]):1]>>[10*:1]',
    
    # 0-1-1: more than one leaving atom, separates into two segments
    '[100*:1]!@[200*,201*,202*:2]~[200*,201*,202*:3]!@[100*:4]>>[*:1][*:2].[*:3][*:4]',
    '[100*:1]!@[200*,201*,202*:2]~[200*,201*,202*:3]~[200*,201*,202*:4]!@[100*:5]>>[*:1][*:2].[*:3][*:4][*:5]',
    '[100*:1]!@[200*,201*,202*:2]'
        +'~[200*,201*,202*;'
        +'$([*][200*,201*,202*]~[200*,201*,202*]!@[100*]),'
        +'$([*][200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]!@[100*]),'
        +'$([*][200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]!@[100*]),'
        +'$([*][200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]!@[100*]),'
        +'$([*][200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]!@[100*]),'
        +'$([*][200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]!@[100*]),'
        +'$([*][200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]~[200*,201*,202*]!@[100*]):3]'
        +'>>[*:1][*:2].[*:3]',
    
    # 0-2. Aromatic ring breaking & other aromaticity changes
    # aromatic ring breaking (leaving atom tagged with isotope 10, needs to be connected to a core structure through an aromatic ring) 
    '[100*;a:1]:[200*,201*,202*;a:2]>>[100*:1]:[10*:2]',
    '[10*;a:1]:[200*,201*,202*;a:2]>>[10*:1]:[10*:2]', #propagate 10*
    '[10*;a:1]=[200*,201*,202*:2]>>[10*:1]=[10*:2]',
    # non-aromatic atom connected to an aromatic atom through a double bond (double bond needed for aromaticity)
    '[100*;a:1]=[200*,201*,202*:2]>>[100*:1]=[10*:2]',

    # Single ring carbon atom leaving if connected to more than 3 core atoms or a carbon core atom
    '[200#6;R;$([#6](@[100#6])@[100*]),$([#6](@[100*])(@[100*])~[100*]);!$([#6]~[200*,201*,202*]):1]>>[10C:1]',

    # Methyl ketone demethylation
    '[100CH0;$([C]=[O;0*,10*,100*]):1]-[200CH3:2]>>[100C:1]-[0CH3:2]',

    # 1,2-diol to aldehyde - total 160 reactions (not worth it?)
    '[OH:1][100CH:2][200CH2:3][201OH:4]>>[OH:1][100CH:2][0CH2:3][0OH:4]',


]


abstraction_smarts = [

    # 1. Ring structures
    # 1-2. Non-aromatic ring
    # carbonyl group - updated version, more specific than the previous one
    '[100CX4:1](-[201S]-[200C])(-[201S]-[200C])>>[1C:1]=[1O]',
    '[100CX4:1](-[100O:2]-[200C])(-[201O,201S]-[200C])>>[1C:1]=[1O:2]',
    '[100CX4:1]1[100O:2]-[200C]-[201O,201S]1>>[1C:1]=[1O:2]',
    
    # ring atom leaving (with one atom between the core atoms) 
    # - if there are two or more, it gets caught by the single atom abstractions
    '[100*;R:1]@[200*;R;A:2]@[100*;R:3]>>[100*:1]-[200*:2].[200*]-[100*:3]',
    '[100*;R:1]@[201*;R;A:2]@[100*;R:3]>>[100*:1]-[201*:2].[201*]-[100*:3]',
    '[100*;R:1]@[202*;R;A:2]@[100*:3]>>[100*:1]-[202*:2].[202*]-[100*:3]',
    '[*;200*,201*,202*;R:1]1@[100*D2;R;A;!#6:2]@[*;200*,201*,202*;R:3]1>>[1*:2]',
    
    # 2. Non-ring structures
    # Single atom abstractions
    # 2-1 Carbon

    '[#6;1*,10*,100*:1]-[200#6]>>[1#6:1]', # neutral carbon
    '[#6;1*,10*,100*:1]=[200#6]>>[2#6:1]', # neutral carbon
    '[#6;1*,10*,100*:1]#[200#6]>>[3#6:1]', # neutral carbon
    '[#6;1*,2*,3*,4*,10*,100*:1]~[201*]>>[4#6:1]', #C+
    '[#6;1*,2*,3*,4*,5*,10*,100*:1]~[202*]>>[5#6:1]', #C-

    # 2-2 Hetero atoms
    '[*;1*,10*,100*;!#6:1]~[200*,201*,202*]>>[1*:1]',
    
    # aromatic n 
    '[nH0;1*,100*:1]~[200*,201*,202*]>>[1nH1:1]',

    
]



