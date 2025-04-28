##################
## 7 # TOPOLOGY ##  -> @TOP <-
##################
from . import IO, FUNC, MAP
import logging, math

ss_names = {
 "F": "Collagenous Fiber",                                                                  #@#
 "E": "Extended structure (beta sheet)",                                                    #@#
 "H": "Helix structure",                                                                    #@#
 "1": "Helix start (H-bond donor)",                                                         #@#
 "2": "Helix end (H-bond acceptor)",                                                        #@#
 "3": "Ambivalent helix type (short helices)",                                              #@#
 "T": "Turn",                                                                               #@#
 "S": "Bend",                                                                               #@#
 "C": "Coil",                                                                               #@#
}

bbss = ss_names.keys()
bbss = FUNC.spl("  F     E     H     1     2     3     T     S     C")  # SS one letter

class martini22:
    ff = True
    def __init__(self):

        # parameters are defined here for the following (protein) forcefields:
        self.name = 'martini22'
        
        # Charged types:
        self.charges = {"Qd":1, "Qa":-1, "SQd":1, "SQa":-1, "RQd":1, "AQa":-1}                                                           #@#
        
        
        #----+---------------------+
        ## A | BACKBONE PARAMETERS |
        #----+---------------------+
        #
        # bbss  lists the one letter secondary structure code
        # bbdef lists the corresponding default backbone beads
        # bbtyp lists the corresponding residue specific backbone beads
        #
        # bbd   lists the structure specific backbone bond lengths
        # bbkb  lists the corresponding bond force constants
        #
        # bba   lists the structure specific angles
        # bbka  lists the corresponding angle force constants
        #
        # bbd   lists the structure specific dihedral angles
        # bbkd  lists the corresponding force constants
        #
        # -=NOTE=- 
        #  if the secondary structure types differ between bonded atoms
        #  the bond is assigned the lowest corresponding force constant 
        #
        # -=NOTE=-
        # if proline is anywhere in the helix, the BBB angle changes for 
        # all residues
        #
        
        ###############################################################################################
        ## BEADS ##                                                                         #                 
        #                              F     E     H     1     2     3     T     S     C    # SS one letter   
        self.bbdef    =    FUNC.spl(" N0   Nda    N0    Nd    Na   Nda   Nda    P5    P5")  # Default beads   #@#
        self.bbtyp    = {                                                                   #                 #@#
                    "ALA": FUNC.spl(" C5    N0    C5    N0    N0    N0    N0    P4    P4"), # ALA specific    #@#
                    "PRO": FUNC.spl(" C5    N0    C5    N0    Na    N0    N0    P4    P4"), # PRO specific    #@#
                    "HYP": FUNC.spl(" C5    N0    C5    N0    N0    N0    N0    P4    P4")  # HYP specific    #@#
        }                                                                                   #                 #@#
        ## BONDS ##                                                                         #                 
        self.bbldef   =             (.365, .350, .310, .310, .310, .310, .350, .350, .350)  # BB bond lengths #@#
        self.bbkb     =             (1250, 1250, None, None, None, None, 1250, 1250, 1250)  # BB bond kB      #@#
        self.bbltyp   = {}                                                                  #                 #@#
        self.bbkbtyp  = {}                                                                  #                 #@#
        ## ANGLES ##                                                                        #                 
        self.bbadef   =             ( 119.2,134,   96,   96,   96,   96,  100,  130,  127)  # BBB angles      #@#
        self.bbka     =             ( 150,   25,  700,  700,  700,  700,   20,   20,   20)  # BBB angle kB    #@#
        self.bbatyp   = {                                                                   #                 #@#
               "PRO":               ( 119.2,134,   98,   98,   98,   98,  100,  130,  127), # PRO specific    #@#
               "HYP":               ( 119.2,134,   98,   98,   98,   98,  100,  130,  127)  # PRO specific    #@#
        }                                                                                   #                 #@#
        self.bbkatyp  = {                                                                   #                 #@#
               "PRO":               ( 150,   25,  100,  100,  100,  100,   25,   25,   25), # PRO specific    #@#
               "HYP":               ( 150,   25,  100,  100,  100,  100,   25,   25,   25)  # PRO specific    #@#
        }                                                                                   #                 #@#
        ## DIHEDRALS ##                                                                     #                 
        self.bbddef   =             ( 90.7,   0, -120, -120, -120, -120)                    # BBBB dihedrals  #@#
        self.bbkd     =             ( 100,   10,  400,  400,  400,  400)                    # BBBB kB         #@#
        self.bbdmul   =             (   1,    1,    1,    1,    1,    1)                    # BBBB mltplcty   #@#
        self.bbdtyp   = {}                                                                  #                 #@#
        self.bbkdtyp  = {}                                                                  #                 #@#
                                                                                            #                 
        ###############################################################################################               
        
        # Some Forcefields use the Ca position to position the BB-bead (me like!)
        # martini 2.1 doesn't
        self.ca2bb = False 
        
        # BBS angle, equal for all ss types                                                         
        # Connects BB(i-1),BB(i),SC(i), except for first residue: BB(i+1),BB(i),SC(i)               
        #                 ANGLE   Ka                                                                
        self.bbsangle =      [   100,  25]                                                               #@#
        
        # Bonds for extended structures (more stable than using dihedrals)                          
        #               LENGTH FORCE                                                                
        self.ebonds   = {                                                                                #@#
               'short': [ .640, 2500],                                                              #@#
               'long' : [ .970, 2500]                                                               #@#
        }                                                                                           #@#
        
        
        #----+-----------------------+
        ## B | SIDE CHAIN PARAMETERS |
        #----+-----------------------+
        
        # To be compatible with Elnedyn, all parameters are explicitly defined, even if they are double.
        self.sidechains = {
            #RES#   BEADS                   BONDS                                                   ANGLES              DIHEDRALS
            #                               BB-SC          SC-SC                                        BB-SC-SC  SC-SC-SC
            "TRP": [FUNC.spl("SC4 SNd SC5 SC5"),[(0.300,5000)]+[(0.270,None) for i in range(5)],        [(210,50),(90,50),(90,50)], [(0,50),(0,200)]],
            "TYR": [FUNC.spl("SC4 SC4 SP1"),    [(0.320,5000), (0.270,None), (0.270,None),(0.270,None)],[(150,50),(150,50)],        [(0,50)]],
            "PHE": [FUNC.spl("SC5 SC5 SC5"),    [(0.310,7500), (0.270,None), (0.270,None),(0.270,None)],[(150,50),(150,50)],        [(0,50)]],
            "HIS": [FUNC.spl("SC4 SP1 SP1"),    [(0.320,7500), (0.270,None), (0.270,None),(0.270,None)],[(150,50),(150,50)],        [(0,50)]],
            "HIH": [FUNC.spl("SC4 SP1 SQd"),    [(0.320,7500), (0.270,None), (0.270,None),(0.270,None)],[(150,50),(150,50)],        [(0,50)]],
            "HSD": [FUNC.spl("SC4 SP1 SQd"),    [(0.320,7500), (0.270,None), (0.270,None),(0.270,None)],[(150,50),(150,50)],        [(0,50)]], # extra
            "ARG": [FUNC.spl("N0 Qd"),          [(0.330,5000), (0.340,5000)],                           [(180,25)]],
            "LYS": [FUNC.spl("C3 Qd"),          [(0.330,5000), (0.280,5000)],                           [(180,25)]],
            "CYS": [FUNC.spl("C5"),             [(0.310,7500)]],
            "ASP": [FUNC.spl("Qa"),             [(0.320,7500)]],
            "GLU": [FUNC.spl("Qa"),             [(0.400,5000)]],
            "ILE": [FUNC.spl("AC1"),            [(0.310,None)]],
            "LEU": [FUNC.spl("AC1"),            [(0.330,7500)]],
            "MET": [FUNC.spl("C5"),             [(0.400,2500)]],
            "ASN": [FUNC.spl("P5"),             [(0.320,5000)]],
            "PRO": [FUNC.spl("C3"),             [(0.300,7500)]],
            "HYP": [FUNC.spl("P1"),             [(0.300,7500)]],
            "GLN": [FUNC.spl("P4"),             [(0.400,5000)]],
            "SER": [FUNC.spl("P1"),             [(0.250,7500)]],
            "THR": [FUNC.spl("P1"),             [(0.260,None)]],
            "VAL": [FUNC.spl("AC2"),            [(0.265,None)]],
            "ALA": [],
            "GLY": [],
            }
        
        # Not all (eg Elnedyn) forcefields use backbone-backbone-sidechain angles and BBBB-dihedrals.
        self.UseBBSAngles          = True 
        self.UseBBBBDihedrals      = True

        # Martini 2.2p has polar and charged residues with seperate charges.
        self.polar   = []
        self.charged = []

        # If masses or charged diverge from standard (45/72 and -/+1) they are defined here.
        self.mass_charge = {
        #RES   MASS               CHARGE
        }

        # Defines the connectivity between between beads
        self.connectivity = {
        #RES       BONDS                                   ANGLES             DIHEDRALS              V-SITE
        "TRP":     [[(0,1),(1,2),(1,3),(2,3),(2,4),(3,4)], [(0,1,2),(0,1,3)], [(0,2,3,1),(1,2,4,3)]],  
        "TYR":     [[(0,1),(1,2),(1,3),(2,3)],             [(0,1,2),(0,1,3)], [(0,2,3,1)]], 
        "PHE":     [[(0,1),(1,2),(1,3),(2,3)],             [(0,1,2),(0,1,3)], [(0,2,3,1)]],
        "HIS":     [[(0,1),(1,2),(1,3),(2,3)],             [(0,1,2),(0,1,3)], [(0,2,3,1)]],
        "HIH":     [[(0,1),(1,2),(1,3),(2,3)],             [(0,1,2),(0,1,3)], [(0,2,3,1)]],
        "HSD":     [[(0,1),(1,2),(1,3),(2,3)],             [(0,1,2),(0,1,3)], [(0,2,3,1)]],# extra
        "GLN":     [[(0,1)]],
        "ASN":     [[(0,1)]],
        "SER":     [[(0,1)]],
        "THR":     [[(0,1)]],
        "ARG":     [[(0,1),(1,2)],                         [(0,1,2)]],
        "LYS":     [[(0,1),(1,2)],                         [(0,1,2)]],
        "ASP":     [[(0,1)]],
        "GLU":     [[(0,1)]],
        "CYS":     [[(0,1)]],
        "ILE":     [[(0,1)]],
        "LEU":     [[(0,1)]],
        "MET":     [[(0,1)]],
        "PRO":     [[(0,1)]],
        "HYP":     [[(0,1)]],
        "VAL":     [[(0,1)]],
        "ALA":     [],
        "GLY":     [],
        }
        
        #----+----------------+
        ## C | SPECIAL BONDS  |
        #----+----------------+
        
        self.special = {
            # Used for sulfur bridges
            # ATOM 1         ATOM 2          BOND LENGTH   FORCE CONSTANT
            (("SC1","CYS"), ("SC1","CYS")):     (0.24,         None),
            }
        
        # By default use an elastic network
        self.ElasticNetwork = False 

        # Elastic networks bond shouldn't lead to exclusions (type 6) 
        # But Elnedyn has been parametrized with type 1.
        self.EBondType = 6
        
        #----+----------------+
        ## D | INTERNAL STUFF |
        #----+----------------+
        
        
        ## BACKBONE BEAD TYPE ##                                                                    
        # Dictionary of default bead types (*D)                                                     
        self.bbBeadDictD  = FUNC.hash(bbss,self.bbdef)                                                            
        # Dictionary of dictionaries of types for specific residues (*S)                            
        self.bbBeadDictS  = dict([(i,FUNC.hash(bbss,self.bbtyp[i])) for i in self.bbtyp.keys()])                        
        
        ## BB BOND TYPE ##                                                                          
        # Dictionary of default abond types (*D)                                                    
        self.bbBondDictD = FUNC.hash(bbss,zip(self.bbldef,self.bbkb))                                                   
        # Dictionary of dictionaries for specific types (*S)                                        
        self.bbBondDictS = dict([(i,FUNC.hash(bbss,zip(self.bbltyp[i],self.bbkbtyp[i]))) for i in self.bbltyp.keys()])       
        # This is tricky to read, but it gives the right bondlength/force constant
        
        ## BBB ANGLE TYPE ##                                                                        
        # Dictionary of default angle types (*D)                                                    
        self.bbAngleDictD = FUNC.hash(bbss,zip(self.bbadef,self.bbka))                                                  
        # Dictionary of dictionaries for specific types (*S)                                        
        self.bbAngleDictS = dict([(i,FUNC.hash(bbss,zip(self.bbatyp[i],self.bbkatyp[i]))) for i in self.bbatyp.keys()])      
                    
        ## BBBB DIHEDRAL TYPE ##                                                                    
        # Dictionary of default dihedral types (*D)                                                 
        self.bbDihedDictD = FUNC.hash(bbss,zip(self.bbddef,self.bbkd,self.bbdmul))                                           
        # Dictionary of dictionaries for specific types (*S)                                        
        self.bbDihedDictS = dict([(i,FUNC.hash(bbss,zip(self.bbdtyp[i],self.bbkdtyp[i]))) for i in self.bbdtyp.keys()])      
        
    # The following function returns the backbone bead for a given residue and                   
    # secondary structure type.                                                                 
    # 1. Look up the proper dictionary for the residue                                          
    # 2. Get the proper type from it for the secondary structure                                
    # If the residue is not in the dictionary of specials, use the default                      
    # If the secondary structure is not listed (in the residue specific                         
    # dictionary) revert to the default.                                                        
    def bbGetBead(self,r1,ss="C"):                                                                   
        return self.bbBeadDictS.get(r1,self.bbBeadDictD).get(ss,self.bbBeadDictD.get(ss))                      
    
    
    def bbGetBond(self,r,a,ss):
        # Retrieve parameters for each residue from table defined above
        b1 = self.bbBondDictS.get(r[0],self.bbBondDictD).get(ss[0],self.bbBondDictD.get(ss[0]))
        b2 = self.bbBondDictS.get(r[1],self.bbBondDictD).get(ss[1],self.bbBondDictD.get(ss[1]))
        # Determine which parameters to use for the bond
        return ( (b1[0]+b2[0])/2, min(b1[1],b2[1]) )
    
    def bbGetAngle(self,r,ca,ss):
        # PRO in helices is dominant
        if r[1] == "PRO" and ss[1] in "H123":
            return self.bbAngleDictS["PRO"].get(ss[1])
        else:
            # Retrieve parameters for each residue from table defined above
            a = [ self.bbAngleDictS.get(r[0],self.bbAngleDictD).get(ss[0],self.bbAngleDictD.get(ss[0])),
                  self.bbAngleDictS.get(r[1],self.bbAngleDictD).get(ss[1],self.bbAngleDictD.get(ss[1])),
                  self.bbAngleDictS.get(r[2],self.bbAngleDictD).get(ss[2],self.bbAngleDictD.get(ss[2])) ]
            # Sort according to force constant
            a.sort(key=lambda i: (i[1],i[0]))
            # This selects the set with the smallest force constant and the smallest angle
            return a[0]
        
    def messages(self):
        '''Prints any force-field specific logging messages.'''
        import logging
        logging.info('Note: Cysteine bonds are 0.24 nm constraints, instead of the published 0.39nm/5000kJ/mol.')

# This is a generic class for Topology Bonded Type definitions
class Bonded:
    # The init method is generic to the bonded types,
    # but may call the set method if atoms are given
    # as (ID, ResidueName, SecondaryStructure) tuples
    # The set method is specific to the different types.
    def __init__(self, other=None, options=None, **kwargs):
        self.atoms      = []
        self.type       = -1
        self.parameters = []
        self.comments   = []
        self.category   = None

        if options and type(options) == dict:
            self.options = options
        if other:
            # If other is given, then copy the attributes
            # if it is of the same class or set the
            # attributes according to the key names if
            # it is a dictionary
            if other.__class__ == self.__class__:
                for attr in dir(other):
                    if not attr[0] == "_":
                        setattr(self, attr, getattr(other, attr))
            elif type(other) == dict:
                for attr in other.keys():
                    setattr(self, attr, other[attr])
            elif type(other) in (list, tuple):
                self.atoms = other

        # For every item in the kwargs keys, set the attribute
        # with the same name. This can be used to specify the
        # attributes directly or to override attributes
        # copied from the 'other' argument.
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # If atoms are given as tuples of
        # (ID, ResidueName[, SecondaryStructure])
        # then determine the corresponding parameters
        # from the lists above
        if self.atoms and type(self.atoms[0]) == tuple:
            self.set(self.atoms, **kwargs)

    def __nonzero__(self):
        return bool(self.atoms)

    def __str__(self):
        if not self.atoms or not self.parameters:
            return ""
        s = ["%5d" % i for i in self.atoms]
        # For exclusions, no type is defined, which equals -1
        if self.type != -1: s.append(" %5d " % self.type)
        # Print integers and floats in proper format and neglect None terms
        s.extend([FUNC.formatString(i) for i in self.parameters if i is not None])
        if self.comments:
            s.append(';')
            if type(self.comments) == str:
                s.append(self.comments)
            else:
                s.extend([str(i) for i in self.comments])
        return " ".join(s)

    def __iadd__(self, num):
        self.atoms = [i + int(num) for i in self.atoms]
        return self

    def __add__(self, num):
        out  = self.__class__(self)
        out += num
        return out

    def __eq__(self, other):
        if type(other) in (list, tuple):
            return self.atoms == other
        else:
            return self.atoms == other.atoms and self.type == other.type and self.parameters == other.parameters

    # This function needs to be overridden for descendents
    def set(self, atoms, **kwargs):
        pass


# The set method of this class will look up parameters for backbone beads
# Side chain bonds ought to be set directly, using the constructor
# providing atom numbers, bond type, and parameters
# Constraints are bonds with kb = None, which can be extracted
# using the category
class Bond(Bonded):
    def set(self, atoms, **kwargs):
        ids, r, ss, ca  = zip(*atoms)
        self.atoms      = ids
        self.type       = 1
        self.positionCa = ca
        self.comments   = "%s(%s)-%s(%s)" % (r[0], ss[0], r[1], ss[1])
        # The category can be used to keep bonds sorted
        self.category   = kwargs.get("category")

        self.parameters = self.options['ForceField'].bbGetBond(r, ca, ss)
        # Backbone bonds also can be constraints.
        # We could change the type further on, but this is more general.
        # Even better would be to add a new type: BB-Constraint
        if self.parameters[1] == None:
            self.category = 'Constraint'

    # Overriding __str__ method to suppress printing of bonds with Fc of 0
    def __str__(self):
        if len(self.parameters) > 1 and self.parameters[1] == 0:
            return ""
        return Bonded.__str__(self)


# Similar to the preceding class
class Angle(Bonded):
    def set(self, atoms, **kwargs):
        ids, r, ss, ca  = zip(*atoms)
        self.atoms      = ids
        self.type       = 2
        self.positionCa = ca
        self.comments   = "%s(%s)-%s(%s)-%s(%s)" % (r[0], ss[0], r[1], ss[1], r[2], ss[2])
        self.category   = kwargs.get("category")
        self.parameters = self.options['ForceField'].bbGetAngle(r, ca, ss)


# Similar to the preceding class
class Vsite(Bonded):
    def set(self, atoms, **kwargs):
        ids, r, ss, ca  = zip(*atoms)
        self.atoms      = ids
        self.type       = 1
        self.positionCa = ca
        self.comments   = "%s" % (r[0])
        self.category   = kwargs.get("category")
        self.parameters = kwargs.get("parameters")


# Similar to the preceding class
class Exclusion(Bonded):
    def set(self, atoms, **kwargs):
        ids, r, ss, ca  = zip(*atoms)
        self.atoms      = ids
        self.positionCa = ca
        self.comments   = "%s" % (r[0])
        self.category   = kwargs.get("category")
        self.parameters = kwargs.get("parameters")


# Similar to the preceding class
class Dihedral(Bonded):
    def set(self, atoms, **kwargs):
        ids, r, ss, ca  = zip(*atoms)
        self.atoms      = ids
        self.type       = 1
        self.positionCa = ca
        self.comments   = "%s(%s)-%s(%s)-%s(%s)-%s(%s)" % (r[0], ss[0], r[1], ss[1], r[2], ss[2], r[3], ss[3])
        self.category   = kwargs.get("category")

        if ''.join(i for i in ss) == 'FFFF':
            # Collagen
            self.parameters = self.options['ForceField'].bbDihedDictD['F']
        elif ''.join(i for i in ss) == 'EEEE' and self.options['ExtendedDihedrals']:
            # Use dihedrals
            self.parameters = self.options['ForceField'].bbDihedDictD['E']
        elif set(ss).issubset("H123"):
            # Helix
            self.parameters = self.options['ForceField'].bbDihedDictD['H']
        else:
            self.parameters = None


# This list allows to retrieve Bonded class items based on the category
# If standard, dictionary type indexing is used, only exact matches are
# returned. Alternatively, partial matching can be achieved by setting
# a second 'True' argument.
class CategorizedList(list):
    def __getitem__(self, tag):
        if type(tag) == int:
            # Call the parent class __getitem__
            return list.__getitem__(self, tag)

        if type(tag) == str:
            return [i for i in self if i.category == tag]

        if tag[1]:
            return [i for i in self if tag[0] in i.category]
        else:
            return [i for i in self if i.category == tag[0]]


class Topology:
    def __init__(self, other=None, options=None, name=""):
        self.name        = ''
        self.nrexcl      = 1
        self.atoms       = CategorizedList()
        self.vsites      = CategorizedList()
        self.exclusions  = CategorizedList()
        self.bonds       = CategorizedList()
        self.angles      = CategorizedList()
        self.dihedrals   = CategorizedList()
        self.impropers   = CategorizedList()
        self.constraints = CategorizedList()
        self.posres      = CategorizedList()
        self.sequence    = []
        self.secstruc    = ""
        # Okay, this is sort of funny; we will add a
        #   #define mapping virtual_sitesn
        # to the topology file, followed by a header
        #   [ mapping ]
        self.mapping     = []
        # For multiscaling we have to keep track of the number of
        # real atoms that correspond to the beads in the topology
        self.natoms      = 0

        if options:
            self.options = options
        else:
            self.options = {}
            self.options['ForceField'] = martini22()
            self.options['PosRes'] = []
        if 'multi' in self.options:
            self.multiscale  = self.options['multi']
        else:
            self.multiscale  = 0
        if not other:
            # Returning an empty instance
            return
        elif isinstance(other, Topology):
            for attrib in ["atoms", "vsites", "bonds", "angles", "dihedrals", "impropers", "constraints", "posres"]:
                setattr(self, attrib, getattr(other, attrib, []))
        elif isinstance(other, IO.Chain):
            if other.type() == "Protein":
                self.fromAminoAcidSequence(other)
            elif other.type() == "Nucleic":
                # Currently there are no Martini Nucleic Acids
                self.fromNucleicAcidSequence(other)
            elif other.type() == "Mixed":
                logging.warning('Mixed Amino Acid /Nucleic Acid chains are not yet implemented')
                # How can you have a mixed chain?
                # Well, you could get a covalently bound lipid or piece of DNA to a protein :S
                # But how to deal with that?
                # Probably one should separate the chains into blocks of specified type,
                # determine the locations of links, then construct the topologies for the
                # blocks and combine them according to the links.
                pass
            else:
                # This chain should not be polymeric, but a collection of molecules
                # For each unique residue type fetch the proper moleculetype
                self.fromMoleculeList(other)
        if name:
            self.name = name

    def __iadd__(self, other):
        if not isinstance(other, Topology):
            other = Topology(other)
        shift     = len(self.atoms)
        last      = self.atoms[-1]
        # The following used work: zip>list expansions>zip back, but that only works if
        # all the tuples in the original list of of equal length. With masses and charges
        # that is not necessarly the case.
        for atom in other.atoms:
            atom = list(atom)
            atom[0] += shift    # Update atom numbers
            atom[2] += last[2]  # Update residue numbers
            atom[5] += last[5]  # Update charge group numbers
            self.atoms.append(tuple(atom))
        for attrib in ["bonds", "vsites", "angles", "dihedrals", "impropers", "constraints", "posres"]:
            getattr(self, attrib).extend([source + shift for source in getattr(other, attrib)])
        return self

    def __add__(self, other):
        out = Topology(self)
        if not isinstance(other, Topology):
            other = Topology(other)
        out += other
        return out

    def __str__(self):
        if self.multiscale:
            out = ['; MARTINI (%s) Multiscale virtual sites topology section for "%s"' % (self.options['ForceField'].name, self.name)]
        else:
            string = '; MARTINI (%s) Coarse Grained topology file for "%s"' % (self.options['ForceField'].name, self.name)
            string += '\n; Created by martinize.py version %s \n; Using the following options:  ' % (self.options['Version'])
            string += ' '.join(self.options['Arguments'])
            out = [string]
        if self.sequence:
            out += [
                '; Sequence:',
                '; ' + ''.join([MAP.AA321.get(AA) for AA in self.sequence]),
                '; Secondary Structure:',
                '; ' + self.secstruc,
                ]

        # Do not print a molecule name when multiscaling
        # In that case, the topology created here needs to be appended
        # at the end of an atomistic moleculetype
        if not self.multiscale:
            out += ['\n[ moleculetype ]',
                    '; Name         Exclusions',
                    '%-15s %3d' % (self.name, self.nrexcl)]

        out.append('\n[ atoms ]')

        # For virtual sites and dummy beads we have to be able to specify the mass.
        # Thus we need two different format strings:
        fs8 = '%5d %5s %5d %5s %5s %5d %7.4f ; %s'
        fs9 = '%5d %5s %5d %5s %5s %5d %7.4f %7.4f ; %s'
        out.extend([len(i) == 9 and fs9 % i or fs8 % i for i in self.atoms])

        # Print out the vsites only if they excist. Right now it can only be type 1 virual sites.
        vsites = [str(i) for i in self.vsites]
        if vsites:
            out.append('\n[ virtual_sites2 ]')
            out.extend(vsites)

        # Print out the exclusions only if they excist.
        exclusions = [str(i) for i in self.exclusions]
        if exclusions:
            out.append('\n[ exclusions ]')
            out.extend(exclusions)

        if self.multiscale:
            out += ['\n;\n; Coarse grained to atomistic mapping\n;',
                    '#define mapping virtual_sitesn',
                    '[ mapping ]']
            for i, j in self.mapping:
                out.append(("%5d     2 " % i)+" ".join(["%5d" % k for k in j]))

            logging.info('Created virtual sites section for multiscaled topology')
            return "\n".join(out)

        # Bonds in order: backbone, backbone-sidechain, sidechain, short elastic, long elastic
        out.append("\n[ bonds ]")
        # Backbone-backbone
        bonds = [str(i) for i in self.bonds["BB"]]
        if bonds:
            out.append("; Backbone bonds")
            out.extend(bonds)
        # Rubber Bands
        bonds = [str(i) for i in self.bonds["Rubber", True]]
        if bonds:
            # Add a CPP style directive to allow control over the elastic network
            out.append("#ifndef NO_RUBBER_BANDS")
            out.append("#ifndef RUBBER_FC\n#define RUBBER_FC %f\n#endif" % self.options['ElasticMaximumForce'])
            out.extend(bonds)
            out.append("#endif")
        # Backbone-Sidechain/Sidechain-Sidechain
        bonds = [str(i) for i in self.bonds["SC"]]
        if bonds:
            out.append("; Sidechain bonds")
            out.extend(bonds)
        # Short elastic/Long elastic
        bonds = [str(i) for i in self.bonds["Elastic short"]]
        if bonds:
            out.append("; Short elastic bonds for extended regions")
            out.extend(bonds)
        bonds = [str(i) for i in self.bonds["Elastic long"]]
        if bonds:
            out.append("; Long elastic bonds for extended regions")
            out.extend(bonds)
        # Cystine bridges
        bonds = [str(i) for i in self.bonds["Cystine"]]
        if bonds:
            out.append("; Cystine bridges")
            out.extend(bonds)
        # Other links
        bonds = [str(i) for i in self.bonds["Link"]]
        if bonds:
            out.append("; Links/Cystine bridges")
            out.extend(bonds)

        # Constraints
        out.append("\n[ constraints ]")
        out.extend([str(i) for i in self.bonds["Constraint"]])

        # Angles
        out.append("\n[ angles ]")
        out.append("; Backbone angles")
        out.extend([str(i) for i in self.angles["BBB"]])
        out.append("; Backbone-sidechain angles")
        out.extend([str(i) for i in self.angles["BBS"]])
        out.append("; Sidechain angles")
        out.extend([str(i) for i in self.angles["SC"]])

        # Dihedrals
        out.append("\n[ dihedrals ]")
        out.append("; Backbone dihedrals")
        out.extend([str(i) for i in self.dihedrals["BBBB"] if i.parameters])
        out.append("; Sidechain improper dihedrals")
        out.extend([str(i) for i in self.dihedrals["SC"] if i.parameters])

        # Postition Restraints
        if self.posres:
            out.append("\n#ifdef POSRES")
            out.append("#ifndef POSRES_FC\n#define POSRES_FC %.2f\n#endif" % self.options['PosResForce'])
            out.append(" [ position_restraints ]")
            out.extend(['  %5d    1    POSRES_FC    POSRES_FC    POSRES_FC' % i for i in self.posres])
            out.append("#endif")

        logging.info('Created coarsegrained topology')
        return "\n".join(out)

    def fromAminoAcidSequence(self, sequence, secstruc=None, links=None,
                              breaks=None, mapping=None, rubber=False,
                              multi=False):
        '''The sequence function can be used to generate the topology for
           a sequence :) either given as sequence or as chain'''

        # Shift for the atom numbers of the atomistic part in a chain
        # that is being multiscaled
        shift = 0
        # First check if we get a sequence or a Chain instance
        if isinstance(sequence, IO.Chain):
            chain         = sequence
            links         = chain.links
            breaks        = chain.breaks
            # If the mapping is not specified, the actual mapping is taken,
            # used to construct the coarse grained system from the atomistic one.
            # The function argument "mapping" could be used to use a default
            # mapping scheme in stead, like the mapping for the GROMOS96 force field.
            mapping       = mapping or chain.mapping
            multi         = chain.multiscale
            self.secstruc = chain.sstypes or len(chain)*"C"
            self.sequence = chain.sequence
            # If anything hints towards multiscaling, do multiscaling
            self.multiscale = self.multiscale or chain.multiscale or multi
            if self.multiscale:
                shift        = self.natoms
                self.natoms += len(chain.atoms())
        elif not secstruc:
            # If no secondary structure is provided, set all to coil
            chain         = None
            self.secstruc = len(self.sequence)*"C"
        else:
            # If a secondary structure is provided, use that. chain is none.
            chain         = None
            self.secstruc = secstruc

        logging.debug(self.secstruc)
        logging.debug(self.sequence)

        # Fetch the sidechains
        # Pad with empty lists for atoms, bonds, angles
        # and dihedrals, and take the first four lists out
        # This will avoid errors for residues for which
        # these are not defined.

        sc = [(self.options['ForceField'].sidechains[res]+5*[[]])[:5] for res in self.sequence]

        # ID of the first atom/residue
        # The atom number and residue number follow from the last
        # atom c.q. residue id in the list processed in the topology
        # thus far. In the case of multiscaling, the real atoms need
        # also be accounted for.
        startAtom = self.natoms + 1
        startResi = self.atoms and self.atoms[-1][2]+1 or 1

        # Backbone bead atom IDs
        bbid = [startAtom]
        for i in [i[0] for i in sc[:-1]]:
            bbid.append(bbid[-1]+len(i)+1)

        # Calpha positions, to get Elnedyn BBB-angles and BB-bond lengths
        # positionCa = [residue[1][4:] for residue in chain.residues]
        # The old method (line above) assumed no hydrogens: Ca would always be
        # the second atom of the residue. Now we look at the name.
        positionCa = []
        for residue in chain.residues:
            for atom in residue:
                if atom[0] == "CA":
                    positionCa.append(atom[4:])

        # Residue numbers for this moleculetype topology
        resid = range(startResi, startResi+len(self.sequence))

        # This contains the information for deriving backbone bead types,
        # bb bond types, bbb/bbs angle types, and bbbb dihedral types and
        # Elnedyn BB-bondlength BBB-angles
        # seqss = zip(bbid, self.sequence, self.secstruc, positionCa)
        seqss = [(bbid[i], self.sequence[i], self.secstruc[i], positionCa[i]) for i in range(len(self.sequence))]

        # Fetch the proper backbone beads
        bb = [self.options['ForceField'].bbGetBead(res, typ) for num, res, typ, Ca in seqss]

        # If termini need to be charged, change the bead types
        if 'NeutralTermini' not in self.options:
            bb[0]  = "Qd"
            bb[-1] = "Qa"

        # If breaks need to be charged, change the bead types
        if 'ChargesAtBreaks' in self.options:
            for i in breaks:
                bb[i]   = "Qd"
                bb[i-1] = "Qa"

        # For backbone parameters, iterate over fragments, inferred from breaks
        for i, j in zip([0]+breaks, breaks+[-1]):
            # Extract the fragment
            frg = j == -1 and seqss[i:] or seqss[i:j]

            # Iterate over backbone bonds
            self.bonds.extend([Bond(pair, category="BB", options=self.options,) for pair in zip(frg, frg[1:])])

            # Iterate over backbone angles
            # Don't skip the first and last residue in the fragment
            self.angles.extend([Angle(triple, options=self.options, category="BBB") for triple in zip(frg, frg[1:], frg[2:])])

            # Get backbone quadruples
            quadruples = zip(frg, frg[1:], frg[2:], frg[3:])

            # No i-1,i,i+1,i+2 interactions defined for Elnedyn
            if self.options['ForceField'].UseBBBBDihedrals:
                # Process dihedrals
                for q in quadruples:
                    id, rn, ss, ca = zip(*q)
                    # Maybe do local elastic networks
                    if ss == ("E", "E", "E", "E") and 'ExtendedDihedrals' not in self.options:
                        # This one may already be listed as the 2-4 bond of a previous one
                        if not (id[0], id[2]) in self.bonds:
                            self.bonds.append(Bond(
                                options    = self.options,
                                atoms      = (id[0], id[2]),
                                parameters = self.options['ForceField'].ebonds['short'],
                                type       = 1,
                                comments   = "%s(%s)-%s(%s) 1-3" % (rn[0], id[0], rn[2], id[2]),
                                category   = "Elastic short"))
                        self.bonds.append(Bond(
                                options    = self.options,
                                atoms      = (id[1], id[3]),
                                parameters = self.options['ForceField'].ebonds['short'],
                                type       = 1,
                                comments   = "%s(%s)-%s(%s) 2-4" % (rn[1], id[1], rn[3], id[3]),
                                category   = "Elastic short"))
                        self.bonds.append(Bond(
                                options    = self.options,
                                atoms      = (id[0], id[3]),
                                parameters = self.options['ForceField'].ebonds['long'],
                                type       = 1,
                                comments   = "%s(%s)-%s(%s) 1-4" % (rn[0], id[0], rn[3], id[3]),
                                category   = "Elastic long"))
                    else:
                        # Since dihedrals can return None, we first collect them separately and then
                        # add the non-None ones to the list
                        dihed = Dihedral(q, options=self.options, category="BBBB")
                        if dihed:
                            self.dihedrals.append(dihed)

            # Elnedyn does not use backbone-backbone-sidechain-angles
            if self.options['ForceField'].UseBBSAngles:
                # Backbone-Backbone-Sidechain angles
                # If the first residue has a sidechain, we take SBB, otherwise we skip it
                # For other sidechains, we 'just' take BBS
                if len(frg) > 1 and frg[1][0]-frg[0][0] > 1:
                    self.angles.append(Angle(
                        options    = self.options,
                        atoms      = (frg[0][0] + 1, frg[0][0], frg[1][0]),
                        parameters = self.options['ForceField'].bbsangle,
                        type       = 2,
                        comments   = "%s(%s)-%s(%s) SBB" % (frg[0][1], frg[0][2], frg[1][1], frg[1][2]),
                        category   = "BBS"))

                # Start from first residue: connects sidechain of second residue
                for (ai, ni, si, ci), (aj, nj, sj, cj), s in zip(frg[0:], frg[1:], sc[1:]):
                    if s[0]:
                        self.angles.append(Angle(
                            options    = self.options,
                            atoms      = (ai, aj, aj+1),
                            parameters = self.options['ForceField'].bbsangle,
                            type       = 2,
                            comments   = "%s(%s)-%s(%s) SBB" % (ni, si, nj, sj),
                            category   = "BBS"))

        # Now do the atom list, and take the sidechains along
        #
        # AtomID AtomType ResidueID ResidueName AtomName ChargeGroup Charge ; Comments
        atid = startAtom
        for resi, resname, bbb, sidechn, ss in zip(resid, self.sequence, bb, sc, self.secstruc):
            scatoms, bon_par, ang_par, dih_par, vsite_par = sidechn

            # Side chain bonded terms
            # Collect bond, angle and dihedral connectivity
            bon_con, ang_con, dih_con, vsite_con = (self.options['ForceField'].connectivity[resname]+4*[[]])[:4]

            # Side Chain Bonds/Constraints
            for atids, par in zip(bon_con, bon_par):
                if par[1] == None:
                    self.bonds.append(Bond(
                        options    = self.options,
                        atoms      = atids,
                        parameters = [par[0]],
                        type       = 1,
                        comments   = resname,
                        category   = "Constraint"))
                else:
                    self.bonds.append(Bond(
                        options    = self.options,
                        atoms      = atids,
                        parameters = par,
                        type       = 1,
                        comments   = resname,
                        category   = "SC"))
                # Shift the atom numbers
                self.bonds[-1] += atid

            # Side Chain Angles
            for atids, par in zip(ang_con, ang_par):
                self.angles.append(Angle(
                    options    = self.options,
                    atoms      = atids,
                    parameters = par,
                    type       = 2,
                    comments   = resname,
                    category   = "SC"))
                # Shift the atom numbers
                self.angles[-1] += atid

            # Side Chain Dihedrals
            for atids, par in zip(dih_con, dih_par):
                self.dihedrals.append(Dihedral(
                    options    = self.options,
                    atoms      = atids,
                    parameters = par,
                    type       = 2,
                    comments   = resname,
                    category   = "SC"))
                # Shift the atom numbers
                self.dihedrals[-1] += atid

            # Side Chain V-Sites
            for atids, par in zip(vsite_con, vsite_par):
                self.vsites.append(Vsite(
                    options    = self.options,
                    atoms      = atids,
                    parameters = par,
                    type       = 1,
                    comments   = resname,
                    category   = "SC"))
                # Shift the atom numbers
                self.vsites[-1] += atid

            # Side Chain exclusions
            # The polarizable forcefield give problems with the charges in the sidechain,
            # if the backbone is also charged.
            # To avoid that, we add explicit exclusions
            if bbb in self.options['ForceField'].charges.keys() and resname in self.options['ForceField'].mass_charge.keys():
                for i in [j for j, d in enumerate(scatoms) if d == 'D']:
                    self.exclusions.append(Exclusion(
                        options    = self.options,
                        atoms      = (atid, i+atid+1),
                        comments   = '%s(%s)' % (resname, resi),
                        parameters = (None, )))

            # All residue atoms
            counter = 0  # Counts over beads
            for atype, aname in zip([bbb] + list(scatoms), MAP.CoarseGrained.residue_bead_names):
                if self.multiscale:
                    atype, aname = "v" + atype, "v" + aname
                # If mass or charge diverse, we adopt it here.
                # We don't want to do this for BB beads because of charged termini.
                if resname in self.options['ForceField'].mass_charge.keys() and counter != 0:
                    M, Q = self.options['ForceField'].mass_charge[resname]
                    aname = Q[counter-1] > 0 and 'SCP' or Q[counter-1] < 0 and 'SCN' or aname
                    self.atoms.append((atid, atype, resi, resname, aname, atid,
                                       Q[counter-1], M[counter-1], ss))
                else:
                    self.atoms.append((atid, atype, resi, resname, aname, atid,
                                       self.options['ForceField'].charges.get(atype, 0), ss))
                # Doing this here save going over all the atoms onesmore.
                # Generate position restraints for all atoms or Backbone beads only.
                if 'all' in self.options['PosRes']:
                    self.posres.append((atid))
                elif aname in self.options['PosRes']:
                    self.posres.append((atid))
                if mapping:
                    self.mapping.append((atid, [i + shift for i in mapping[counter]]))
                atid    += 1
                counter += 1

        # The rubber bands are best applied outside of the chain class, as that gives
        # more control when chains need to be merged. The possibility to do it on the
        # chain level is retained to allow building a complete chain topology in
        # a straightforward manner after importing this script as module.
        if rubber and chain:
            rubberList = rubberBands(
                [(i[0], j[4:7]) for i, j in zip(self.atoms, chain.cg()) if i[4] in ElasticBeads],
                ElasticLowerBound, ElasticUpperBound,
                ElasticDecayFactor, ElasticDecayPower,
                ElasticMaximumForce, ElasticMinimumForce)
            self.bonds.extend([Bond(i, options=self.options, type=6,
                                    category="Rubber band") for i in rubberList])

        # Note the equivalent of atomistic atoms that have been processed
        if chain and self.multiscale:
            self.natoms += len(chain.atoms())

    def fromNucleicAcidSequence(self, sequence, secstruc=None, links=None, breaks=None,
                                mapping=None, rubber=False, multi=False):

        # Shift for the atom numbers of the atomistic part in a chain
        # that is being multiscaled
        shift = 0
        # First check if we get a sequence or a Chain instance
        if isinstance(sequence, IO.Chain):
            chain         = sequence
            links         = chain.links
            breaks        = chain.breaks
            # If the mapping is not specified, the actual mapping is taken,
            # used to construct the coarse grained system from the atomistic one.
            # The function argument "mapping" could be used to use a default
            # mapping scheme in stead, like the mapping for the GROMOS96 force field.
            mapping       = mapping or chain.mapping
            multi         = self.options['multi'] or chain.multiscale
            self.secstruc = chain.sstypes or len(chain)*"C"
            self.sequence = chain.sequence
            # If anything hints towards multiscaling, do multiscaling
            self.multiscale = self.multiscale or chain.multiscale or multi
            if self.multiscale:
                shift        = self.natoms
                self.natoms += len(chain.atoms())
        elif not secstruc:
            # If no secondary structure is provided, set all to coil
            chain         = None
            self.secstruc = len(self.sequence)*"C"
        else:
            # If a secondary structure is provided, use that. chain is none.
            chain         = None
            self.secstruc = secstruc

        logging.debug(self.secstruc)
        logging.debug(self.sequence)

        # Fetch the base information
        # Pad with empty lists for atoms, bonds, angles
        # and dihedrals, and take the first five lists out
        # This will avoid errors for residues for which
        # these are not defined.
        sc = [(self.options['ForceField'].bases[res]+6*[[]])[:6] for res in self.sequence]

        # ID of the first atom/residue
        # The atom number and residue number follow from the last
        # atom c.q. residue id in the list processed in the topology
        # thus far. In the case of multiscaling, the real atoms need
        # also be accounted for.
        startAtom = self.natoms + 1
        startResi = self.atoms and self.atoms[-1][2]+1 or 1

        # Backbone bead atom IDs
        bbid = [[startAtom, startAtom+1, startAtom+2]]
        for i in zip(*sc)[0]:
            bbid1 = bbid[-1][0]+len(i)+3
            bbid.append([bbid1, bbid1+1, bbid1+2])

        # Residue numbers for this moleculetype topology
        resid = range(startResi, startResi+len(self.sequence))

        # This contains the information for deriving backbone bead types,
        # bb bond types, bbb/bbs angle types, and bbbb dihedral types.
        seqss = [(bbid[i], self.sequence[i], self.secstruc[i]) for i in range(len(self.sequence))]

        # Fetch the proper backbone beads
        # Since there are three beads we need to split these to the list
        bb = [self.options['ForceField'].bbGetBead(res, typ) for num, res, typ in seqss]
        bb3 = [i for j in bb for i in j]

        # This is going to be usefull for the type of the last backbone bead.
        # If termini need to be charged, change the bead types
        # if not self.options['NeutralTermini']:
        #    bb[0]  ="Qd"
        #    bb[-1] = "Qa"

        # If breaks need to be charged, change the bead types
        # if self.options['ChargesAtBreaks']:
        #    for i in breaks:
        #        bb[i]   = "Qd"
        #        bb[i-1] = "Qa"

        # For backbone parameters, iterate over fragments, inferred from breaks
        for i, j in zip([0]+breaks, breaks+[-1]):
            # Extract the fragment
            frg = j == -1 and seqss[i:] or seqss[i:j]
            # Expand the 3 bb beads per residue into one long list
            # Resulting list contains three tuples per residue
            # We use the useless ca parameter to get the correct backbone bond from bbGetBond
            frg = [(j[0][i], j[1], j[2], i) for j in frg for i in range(len(j[0]))]

            # Iterate over backbone bonds
            self.bonds.extend([Bond(pair, category="BB", options=self.options,) for pair in zip(frg, frg[1:])])

            # Iterate over backbone angles
            # Don't skip the first and last residue in the fragment
            self.angles.extend([Angle(triple, options=self.options, category="BBB") for triple in zip(frg, frg[1:], frg[2:])])

            # Get backbone quadruples
            quadruples = zip(frg, frg[1:], frg[2:], frg[3:])

            # No i-1,i,i+1,i+2 interactions defined for Elnedyn
            # Process dihedrals
            for q in quadruples:
                id, rn, ss, ca = zip(*q)
                # Since dihedrals can return None, we first collect them separately and then
                # add the non-None ones to the list
                dihed = Dihedral(q, options=self.options, category="BBBB")
                if dihed:
                    self.dihedrals.append(dihed)

        # Now do the atom list, and take the sidechains along
        #
        atid  = startAtom
        # We need to do some trickery to get all 3 bb beads in to these lists
        # This adds each element to a list three times, feel free to shorten up
        resid3 = [i for i in resid for j in range(3)]
        sequence3 = [i for i in self.sequence for j in range(3)]
        sc3 = [i for i in sc for j in range(3)]
        secstruc3 = [i for i in self.secstruc for j in range(3)]
        count = 0
        for resi, resname, bbb, sidechn, ss in zip(resid3, sequence3, bb3, sc3, secstruc3):
            # We only want one side chain per three backbone beads so this skips the others
            if (count % 3) == 0:
                # Note added impropers in contrast to aa
                scatoms, bon_par, ang_par, dih_par, imp_par, vsite_par = sidechn

                # Side chain bonded terms
                # Collect bond, angle and dihedral connectivity
                # Impropers needed to be added here for DNA
                bon_con, ang_con, dih_con, imp_con, vsite_con = (self.options['ForceField'].connectivity[resname]+5*[[]])[:5]

                # Side Chain Bonds/Constraints
                for atids, par in zip(bon_con, bon_par):
                    if par[1] == None:
                        self.bonds.append(Bond(
                            options    = self.options,
                            atoms      = atids,
                            parameters = [par[0]],
                            type       = 1,
                            comments   = resname,
                            category   = "Constraint"))
                    else:
                        self.bonds.append(Bond(
                            options    = self.options,
                            atoms      = atids,
                            parameters = par,
                            type       = 1,
                            comments   = resname,
                            category   = "SC"))
                    # Shift the atom numbers
                    self.bonds[-1] += atid

                # Side Chain Angles
                for atids, par in zip(ang_con, ang_par):
                    self.angles.append(Angle(
                        options    = self.options,
                        atoms      = atids,
                        parameters = par,
                        type       = 2,
                        comments   = resname,
                        category   = "SC"))
                    # Shift the atom numbers
                    self.angles[-1] += atid

                # Side Chain Dihedrals
                for atids, par in zip(dih_con, dih_par):
                    self.dihedrals.append(Dihedral(
                        options    = self.options,
                        atoms      = atids,
                        parameters = par,
                        type       = 1,
                        comments   = resname,
                        category   = "BSC"))
                    # Shift the atom numbers
                    self.dihedrals[-1] += atid

                # Side Chain Impropers
                for atids, par in zip(imp_con, imp_par):
                    self.dihedrals.append(Dihedral(
                        options    = self.options,
                        atoms      = atids,
                        parameters = par,
                        type       = 2,
                        comments   = resname,
                        category   = "SC"))
                    # Shift the atom numbers
                    self.dihedrals[-1] += atid

                # Side Chain V-Sites
                for atids, par in zip(vsite_con, vsite_par):
                    self.vsites.append(Vsite(
                        options    = self.options,
                        atoms      = atids,
                        parameters = par,
                        type       = 1,
                        comments   = resname,
                        category   = "SC"))
                    # Shift the atom numbers
                    self.vsites[-1] += atid

                # Currently DNA needs exclusions for the base
                # The loop runs over the first backbone bead so 3 needs to be added to the indices
                for i in range(len(scatoms)):
                    for j in range(i+1, len(scatoms)):
                        self.exclusions.append(Exclusion(
                            options    = self.options,
                            atoms      = (i+atid+3, j+atid+3),
                            comments   = '%s(%s)' % (resname, resi),
                            parameters = (None, )))

                # All residue atoms
                counter = 0  # Counts over beads
                # Need to tweak this to get all the backbone beads to the list with the side chain
                bbbset = [bb3[count], bb3[count+1], bb3[count+2]]
                for atype, aname in zip(bbbset+list(scatoms), MAP.CoarseGrained.residue_bead_names_dna):
                    if self.multiscale:
                        atype, aname = "v"+atype, "v"+aname
                    self.atoms.append((atid, atype, resi, resname, aname, atid,
                                       self.options['ForceField'].charges.get(atype, 0), ss))
                    # Doing this here saves going over all the atoms onesmore.
                    # Generate position restraints for all atoms or Backbone beads only.
                    if 'all' in self.options['PosRes']:
                        self.posres.append((atid))
                    elif aname in self.options['PosRes']:
                        self.posres.append((atid))
                    if mapping:
                        self.mapping.append((atid, [i+shift for i in mapping[counter]]))
                    atid    += 1
                    counter += 1
            count += 1

        # One more thing, we need to remove dihedrals (2) and an angle (1)  that reach beyond the 3' end
        # This is stupid to do now but the total number of atoms seems not to be available before
        # This iterate the list in reverse order so that removals don't affect later checks
        for i in range(len(self.dihedrals)-1, -1, -1):
            if (max(self.dihedrals[i].atoms) > self.atoms[-1][0]):
                del self.dihedrals[i]
        for i in range(len(self.angles)-1, -1, -1):
            if (max(self.angles[i].atoms) > self.atoms[-1][0]):
                del self.angles[i]

    def fromMoleculeList(self, other):
        pass