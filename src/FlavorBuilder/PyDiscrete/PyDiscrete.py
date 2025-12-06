import numpy as np
import sympy as sp
import os
import itertools
import subprocess
import ast
import re
from pathlib import Path
import importlib.resources as ir
import pickle

class Group():
 
    """
     This class represents a discrete group obtained by GAP. The functions in this class can compute group theory objects such as CG coefficients or tensor products. You can also define fields which transform under the different irreducible representations of the group. Basic information from the group is obtained through GAP but this is stored in this class too.
     
     Attributes
     ----------
     
       - name: str
           This is the name of the group such that GAP can read it. It is always of the form "SmallGroup(gapID1, gapID2)". Here gapID1 and gapID2 are integers used only during initialization.
        
       - GAPoutput: str
           Output obtained from calling GAP. It shows the basic information such as the character table, matrix representations and size of conjugacy classes.
        
       - GAPerror: str
           Error printed by GAP in case the group was not called.

       -  GAP_run : boolean
           Boolean that tells whether GAP was used or the group was taken from the saved_groups.pkl file.
    
       - structure_description: str
           Name given by GAP.
       
       - size: int
           Size of the group.
       
       - rep_matrices: list
           list of the irreducbible representation matrices of the group. The matrices are numpy arrays.
       
       - dimensions_of_reps: list
           Contains the dimensions of the irreducible representations of the group.
    
       - number_of_reps: int
           number of irreducible representaations of the group.
    
       - size_conjugacy_classes : list
           Sizes of the conjugacy classes of the group.
    
       - character_table: numpy array
          Character tabel of the group.
      
       - unitary_representations: bool
          If True, the group has all unitary representations. This is computed by PyDiscrete when the  Group class is initialized.
        
       - cg_coeff_calculated: numpy array
          Contains boolean values of the CG coeffficients calculated for the different products of the group. It is initialized as a matrix of zeroes.
        
       - cg_coeff: numpy array
          Contains the float CG coefficients for the different products of the group.
        
       - kronecker_product_calculated: numpy array
          Contains boolean values of whether the kronecker products (multiplicities of the sum of irreducible representations obtained from the product of two irreps) have been previously calculated.
        
       - kronecker_product: numpy array
          Contains the kronecker products (multiplicities of the irreducible representations obtained from the product of two irreps).

       
    
    
    Methods:
    --------
    
    
       - isUnitary():
          Checks if a group is unitary
        
       - isUnitaryRep():
          Checks if a irrep is unitary
        
       - getCGC(p,q):
          Returns a numpy array with the CG coefficients for the product  of the p-th irrep times the q-th irrep. The ordering of p,q is based on the order of the character table
        
       - getCGCcc(p,q):
          Returns the complex conjugate transpose  of the array given by getCGC, if the group is unitary. Otherwise, it gives the inverse of the array given by getCGC
        
       - getKroneckerProduct(p,q):
          Returns a numpy array of the multiplicity of the irreducible representations obtained in the decomposition of the product of the pth irrep and the qth irrep
        
       - getMultiplicity(p,q,r):
          Returns an integer which corresponds to the multiplicity of the r-th irrep in the product of the p-th and q-th irreps.
        
       - getRepVector(pRep, label_p):
          Returns a list with vectors of sympy symbols using label_p as a lable. This list corresponds to a field transforming in the pRep-th irrep.
        
       - multiply(list_fields):
          Returns a list that contains the irrep decomposition of the product of the fields in the list_fields
        
       - extractOneRep(vec_rep, idx_p):
          Returns a vector with only the idx_p-th irrep in the list of irreps in vec_rep

    """

   

    def __init__(self, gapID1, gapID2, gap_path, compute_ALL_CGC = True):
        """
        
        Input:
        ----------
        
          - gapID1: integer
             First number of the GAP ID
        
          - gapID2: integer
             Second number of the GAP ID
        
          - gap_path: str
             Path for the location of GAP

          - compute_ALL_CGC: boolean, optional
             If True, it computes all CGC coefficients when initialized
        
        
        """
        self.name = "SmallGroup(" + str(gapID1) + "," + str(gapID2) + ")"

        
        with ir.as_file(
                ir.files("FlavorBuilder.PyDiscrete") / "saved_groups.pkl"
        ) as saved_groups_pickle_path:
            with open(saved_groups_pickle_path, "rb") as f:
                self.saved_groups_dict = pickle.load(f)

        if (gapID1, gapID2) in self.saved_groups_dict:
            self.GAPoutput = self.saved_groups_dict[(gapID1, gapID2)]
            self.GAP_run = False

        else:
            # Path for .gap file
            gap_exe = Path(gap_path).expanduser().resolve()
            if not gap_exe.exists() or not os.access(gap_exe, os.X_OK):
                raise RuntimeError(
                    f"Invalid GAP executable path: '{gap_exe}'. "
                    "Provide the full path to the GAP binary and ensure it is executable."
                    )

            #Prepare gap script to extract group info
            with ir.as_file(
                    ir.files("FlavorBuilder.PyDiscrete") / "GAP-Script" / "extractGroupDataPy.gap"
            ) as gap_script_path:
                    # Make a GAP-safe literal path (forward slashes; escape quotes)
                    gap_script_literal = '"' + gap_script_path.as_posix().replace('"', r'\"') + '"'


                    # 3) Build batch code exactly like your original logic
                    gap_code = (
                        f"Read({gap_script_literal});; "
                        f"MBgetGroupInfo({self.name});; "
                        f"quit;;"
                        )


            # Run GAP as a subprocess
            result = subprocess.run(
                [str(gap_exe), "-q", "-b"],
                input=gap_code.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.GAPoutput = result.stdout.decode("utf-8", "replace").strip()
            self.GAPerror = result.stderr.decode("utf-8", "replace").strip()
        
            #Prints the GAP error if the group properties were not sucesfully obtained from GAP
            if result.returncode != 0:
                raise RuntimeError(
                    f"GAP execution failed for {self.name} (exit code {result.returncode}).\n"
                    f"STDERR:\n{self.GAPerror}"
                )

            if self.GAPerror:
                print(f"[Warning] GAP reported messages for {self.name}:\n{self.GAPerror}\n")
            else:
                self.saved_groups_dict[(gapID1, gapID2)] = self.GAPoutput
        

        #Looks for the information in the GAPoutput
        match_StructureDescription = re.search(r"StructureDescription:\s*(.*?)\s*Size:",self.GAPoutput,re.DOTALL)
        match_Size = re.search(r"Size:\s*(.*?)\s*DimensionsOfReps:", self.GAPoutput, re.DOTALL)
        match_RepMatrices = re.search(r"RepMatrices:\s*(.*)", self.GAPoutput, re.DOTALL)
        match_DimensionsOfReps = re.search(r"DimensionsOfReps:\s*(.*?)\s*SizesConjugacyClasses:", self.GAPoutput, re.DOTALL)
        match_SizeConjugacyClasses = re.search(r"SizesConjugacyClasses:\s*(.*?)\s*CharacterTable:", self.GAPoutput, re.DOTALL)
        match_CharacterTable = re.search(r"CharacterTable:\s*(.*?)\s*RepMatrices:", self.GAPoutput, re.DOTALL)


        #Replace GAP output for actual numerical values
            
        # First replace E(n)^k (must be done before the plain E(n) case)
        s_RepMatrices = re.sub(r"E\((\d+)\)\^(\d+)", self._replace_E_power, match_RepMatrices.group(1))
        s_CharacterTable = re.sub(r"E\((\d+)\)\^(\d+)", self._replace_E_power, match_CharacterTable.group(1))

        # Then replace plain E(n)
        s_RepMatrices = re.sub(r"E\((\d+)\)", self._replace_E_plain, s_RepMatrices)
        s_CharacterTable = re.sub(r"E\((\d+)\)", self._replace_E_plain, s_CharacterTable)



        # Use eval, but restrict to only allow numpy functions
        safe_globals = {"np": np}
        nested_RepMatrices = eval(s_RepMatrices, safe_globals)
        nested_CharacterTable = eval(s_CharacterTable, safe_globals)



        # Convert nested list into list of numpy arrays
        final_RepMatrices = [[np.array(entry) for entry in row] for row in nested_RepMatrices]
        final_CharacterTable = np.array([[entry for entry in row] for row in nested_CharacterTable])

        
        
        #Define the attributes of the group class
        self.structure_description = match_StructureDescription.group(1).replace('"', '')
        self.size = int(match_Size.group(1))
        self.rep_matrices = final_RepMatrices
        self.dimensions_of_reps = ast.literal_eval(match_DimensionsOfReps.group(1))
        self.number_of_reps = len(ast.literal_eval(match_DimensionsOfReps.group(1)))
        self.size_conjugacy_classes = ast.literal_eval(match_SizeConjugacyClasses.group(1))
        self.character_table = final_CharacterTable
        self.unitary_representations = self.isUnitary()
        self.cg_coeff_calculated = np.zeros((self.number_of_reps, self.number_of_reps), dtype=bool)
        self.cg_coeff = np.zeros((self.number_of_reps, self.number_of_reps), dtype=object)
        self.kronecker_product_calculated = np.zeros((self.number_of_reps, self.number_of_reps), dtype=bool)
        self.kronecker_product = np.zeros((self.number_of_reps, self.number_of_reps), dtype=object)
        self.NumberOfSinglets = np.count_nonzero(np.array(self.dimensions_of_reps) == 1)

        # If compute_ALL_CGC = True, then all CG coefficients are computed when the group is initialized
        if compute_ALL_CGC:
            for li in range(1, self.number_of_reps + 1):
                for lj in range(1, self.number_of_reps + 1):
                    self.getCGC(li, lj)
                
                
                
    def isUnitary(self):
    
        """
        Checks if the group is unitary.
        
        Output:
        -------
        
          - unitaryQ:Boolean.
              If true, the group is unitary.
        """
        
        num_reps = self.number_of_reps
        unitaryQ = True
        for li in range(num_reps):
            lli = li +1
            if not self.isUnitaryRep(lli):
                unitaryQ = False
                return unitaryQ
        return unitaryQ
    
    def isUnitaryRep(self,rep_idx_P):
    
        """
        Checks whether the specified irrep is unitary. The irreducible representations are labeled according to their order in the character table obtained from GAP. The function tests if the `rep_idx_P`-th representation is unitarty.

        
        Input:
        ----------
          - rep_idx_P: integer
               Index of the irrep (1-based) whose unitarity is to be checked. The ordering is based in the character table obtained from GAP.
           
        Output:
        -------
           - unitaryrepQ: Boolean.
               If true, the rep_idx_P-th irrep is unitary. False otherwise.
        
        """
        
        if not (1 <= rep_idx_P <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: rep_idx_P = {rep_idx_P}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        
        rep_idx_P = rep_idx_P -1
        #Checks if all matrices in the irrep rep_idx_P satisfy U^{\dagger} U = I
        rep_matrices = self.rep_matrices[rep_idx_P]
        products = [np.conj(U.T) @ U for U in rep_matrices]
        simplified = [np.round(P, decimals=6) for P in products]
        unique_matrices = {tuple(map(tuple, P)) for P in simplified}
        if len(unique_matrices) != 1:
            unitary_representationsREP = False
        
        identity = (np.eye(rep_matrices[0].shape[0]))
        
        unitaryrepQ = np.allclose(list(unique_matrices)[0], identity)

        return unitaryrepQ
    
    
    def _replace_E_power(self, some_gap_str):
    
        """
        This is a helper function. It replaces  E(n)^k for np.exp(2j * np.pi * {k} / {n}) in the GAP output.
        
        Input:
        -----------
        
           - some_gap_str: str
               String obtained from GAP output.
            
        
        Output:
        -------
        
            - repstr: str.
               A string where E(n)^k gets replaced by np.exp(2j * np.pi * {k} / {n}).
        """
        
        n, k = some_gap_str.group(1), some_gap_str.group(2)
        
        repstr = f"np.exp(2j * np.pi * {k} / {n})"
        return repstr

    def _replace_E_plain(self, some_gap_str):

        """
        This is a helper function. It replaces  E(n) for np.exp(2j * np.pi  / {n}) in the GAP output.
        
        Input:
        -----------
        
            - some_gap_str: str
               String obtained from GAP output.
            
        
        Output:
        -------
        
            - repstr: str.
               A string where E(n) gets replaced by np.exp(2j * np.pi  / {n}).
        """
        
        n = some_gap_str.group(1)
        repstr = f"np.exp(2j * np.pi / {n})"
        return repstr

    def _mIndex(self, dim_q, li, lj):
        """
        Helper function to calculate the CG coefficients.
        
        Parmeters
        ---------
        
            - dim_q: int.
                Dimension of second irrep q. The irreducible representations are labeled according to their order in the character table obtained from GAP.
           
            - li: int.
                Index for the first irrep p. Must be between 0 and the sp (dimension of p irrep).
           
            - lj: int.
                Index for the second irrep q (whose dimension it not a parameter in this function). Must be between 0 and the dim_q (dimension of q irrep).
           
        Output:
        -------
        
            - mindex: int.
                Integer given by (li-1) * dim_q  + lj.
    
        
        """
        
        mindex = (li - 1) * dim_q + lj
        return mindex

    def _pickTuple(self, dim_r, dim_p, dim_q, Amat, plstart=0):
    
        """
        Checks if one of the entries in the diagonal of the matrix Amat is non-zero. This is a helper function to construct the CG coeffcients.
        
        
        Input:
        ----------
        
            - dim_r: int
                Dimension of r irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - dim_p: int
                Dimension of p irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - dim_q: int
                Dimension of q irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - Amat: numpy array
                Numpy array with matrices obtained from the helper function _A() from which can be obtained all the CG coefficients.
            - plstart: int, optional
                Index for where to start looking in the diagonal. Default is zero. If zero, then it looks over all the diagonal.
           
           
        Output:
        -------
            - ptuple: list of integers.
                List of integers if one of the entries in the diagonal of the matrix Amat is non-zero. Otherwise it returns none.
           
        """
        lstatus = True
        ll = plstart

        while lstatus and ll < dim_r:
            ll += 1
            li = 0
            
            while lstatus and li < dim_p:
                li += 1
                lj = 0
                
                while lstatus and lj < dim_q:
                    lj += 1
                    if Amat[ll-1][ll-1][self._mIndex(dim_q, li, lj) - 1, self._mIndex(dim_q, li, lj) - 1] != 0:
                        lstatus = False

        if lstatus:
            ptuple = None
            return ptuple
        else:
            ptuple = (li, lj, ll)
            return ptuple

    def _VectorZeroQ(self, vec):
    
        """
        Checks if a vector is non-zero
        
        Input:
        -----------
        
            - vec: numpy array
                The vector to be tested if it is non-zero.
            
            
        Output:
        -------
        
            - zeroQ: Boolean.
                It is True if vec is a non-zero vector.
        
        """
        lLen = len(vec)
        
        if lLen == 0:
            zeroQ = False
            return zeroQ
        
        if np.array_equal(vec, np.zeros(lLen)):
            zeroQ = True
            return zeroQ
        else:
            zeroQ = False
            return zeroQ

    def _getC(self, dim_r, dim_p, dim_q, l, Amat, plstart=0):
    
        """
        This is a helper function to calculate the CG coefficients.
        
        Input:
        ----------
        
            - dim_r: int
                 Dimension of the r irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - dim_p: int
                 Dimension of the p irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - dim_q: int
                Dimension of the q irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - l: int
                Number between 0 and dim_r.
            - Amat: numpy array
                Numpy array with matrices obtained from the helper function _A() from which can be obtained all the CG coefficients.
            - plstart: int, optional
                Index for where to start looking for non-zero entries in the diagonal of the matrix Amat. Default is zero. If zero, then it looks over all the diagonal.
           
           
        Output:
        -------
        
            - getCarray: numpy array.
                A numpy array with specific elements from the matrix Amat. This will help obtain the CG coefficients to get the q-th irrep from the product of the r-th and p-th irrep.
        """
        
        li, lj, lk = self._pickTuple(dim_r, dim_p, dim_q, Amat, plstart)
        
        result = []
        for m in range(1, dim_p + 1):
            for n in range(1, dim_q + 1):
                result.append(
                    Amat[l-1][lk-1][self._mIndex(dim_q, m, n) - 1, self._mIndex(dim_q, li, lj) - 1]
                )
                
        getCarray = np.array(result).flatten()
        
        return getCarray


    def _getCNormalized(self, dim_r, dim_p, dim_q, Amat):
    
        """
        This is a helper function to calculate the CG coefficients when the group is unitary.
        
        Input:
        ----------
        
            - dim_r: int
                Dimension of the r irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - dim_p: int
                Dimension of the p irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - dim_q: int
                Dimension of the q irrep. The irreducible representations are labeled according to their order in the character table obtained from GAP.
            - Amat: numpy array
                Numpy array with matrices obtained from the helper function _A() from which can be obtained all the CG coefficients
           
           
        Output:
        -------
        
            - getCarraynom: numpy array.
                A numpy array with specific elements from the matrix Amat. This will help obtain the CG coefficients to get the q-th irrep from the product of the r-th and p-th irrep.
        """
    
        
        li, lj, lk = self._pickTuple(dim_r, dim_p, dim_q, Amat)
        result = []
        for ll in range(1, dim_r + 1):
            flattened_list = []
            for m in range(1, dim_p + 1):
                for n in range(1, dim_q + 1):
                    flattened_list.append(
                        Amat[ll-1][lk-1][self._mIndex(dim_q, m, n) - 1, self._mIndex(dim_q, li, lj) - 1]
                    )
            result.append(np.array(flattened_list))
        
        normalization_factor = np.sqrt(
            Amat[lk-1][lk-1][self._mIndex(dim_q, li, lj) - 1, self._mIndex(dim_q, li, lj) - 1]
        )
        
        getCarraynom = np.array(result) / normalization_factor
        
        return getCarraynom


    def _A(self, idx_r, idx_p, idx_q, pl, pk):
    
        """
        Calculates the matrix A used to obtain the CG coefficients. This is a helper function.
        
        Input:
        ----------
        
            - idx_r: int
                Index (1-based) of the r irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
            - pl: int
               index between 0 and the dimension of the r irrep.
            - pk: int
                index between 0a nd the dimension of the k irrep.
        
        Output:
        -------
        
            - matrixA: a numpy array.
                A numpy array with the matrix A which will be usted to get the CG coefficients.
        """
        def compute_sum(T, m, i, n, j): #Computes the summand for eq. (9) in Van den Broek, P. M., & Cornwell, J. F. (1978)
            return (
                self.rep_matrices[idx_p-1][T-1][m-1][i-1] *
                self.rep_matrices[idx_q-1][T-1][n-1][j-1] *
                np.conjugate(self.rep_matrices[idx_r-1][T-1][pl-1][pk-1])
                   )

        def compute_inner(m, n, i, j): #Computes the sum of eq. (9) in Van den Broek, P. M., & Cornwell, J. F. (1978)
            return sum(
                compute_sum(T, m, i, n, j)
                for T in range(1, self.size + 1)
            ) * self.dimensions_of_reps[idx_r-1] / self.size


        result = np.array([
                [
                    np.array([
                        [
                            compute_inner(m, n, i, j)
                            for j in range(1, self.dimensions_of_reps[idx_q-1] + 1)
                        ]
                        for i in range(1, self.dimensions_of_reps[idx_p-1] + 1)
                    ]).flatten()
                    for n in range(1, self.dimensions_of_reps[idx_q-1] + 1)
                ]
                for m in range(1, self.dimensions_of_reps[idx_p-1] + 1)
            ])
        matrixA = np.array([item for sublist in result for item in sublist])
        return matrixA


    def _generateKroneckerProduct(self, idx_p, idx_q):
    
        """
        Calculates a numpy array with the multiplicity of the irreducible representations obtained in the decomposition of the product of the p irrep and the q irrep.
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
          
        Output:
        -------
        
            - kroneckerarray: numpy array.
                A numpy array with the multiplicities of all the irreps in the product of the irreps p x q.
        
        """
        
        # Extract the character table
        character_table = self.character_table
        size_conjugacy_classes = np.array(self.size_conjugacy_classes)
        group_size = self.size

        #Calculate multiplicites from character formula
        product = character_table[ idx_p-1,:] * character_table[idx_q-1,:] * size_conjugacy_classes
        result = np.dot(character_table, np.conjugate(product)) / group_size
        complex_array = result
        
        
        #Round up to threshold
        threshold = 1e-5

        # Chop small real parts
        real_part = np.real(complex_array)
        real_part[np.abs(real_part) < threshold] = 0

        # Chop small imaginary parts
        imag_part = np.imag(complex_array)
        imag_part.setflags(write=True)
        imag_part[np.abs(imag_part) < threshold] = 0

        # Reconstruct the complex array with chopped parts
        kroneckerarray = real_part + 1j * imag_part
        
        

        return kroneckerarray


    def _CGC(self, idx_p, idx_q):
    
        """
        Calculates the CG coefficients for the product of the p  irrep times the q irrep. The ordering (1-based) is based in the character table obtained from GAP.
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
          
          
        Output:
        -------
        
            - CGCmatrix: numpy array.
                This numphy array contains the CG coefficients.
          
        """
        sp = self.dimensions_of_reps[idx_p-1]
        sq = self.dimensions_of_reps[idx_q-1]
        U = []
        
        
        #Calculate the CG coefficients from the matrix A()
        for idx_r in range(1, self.number_of_reps + 1):
            sr = self.dimensions_of_reps[idx_r-1]
            lmulti = self.getMultiplicity(idx_p, idx_q, idx_r)
            if lmulti == 0:
                continue

            lA = np.array([[self._A(idx_r, idx_p, idx_q, l+1, k+1) for k in range(sr)] for l in range(sr)])
            lk = 0
            ln = 1
            
            while ln <= lmulti:
                lc = np.array([np.array(self._getC(sr, sp, sq, ll+1, lA, lk)) for ll in range(sr)])
                U.append(lc)
                ln += 1

                li, lj, lk = self._pickTuple(sr, sp, sq, lA, lk)
                lk += 1
                
        U2 = np.array([item for sublist in np.array(U, dtype=object) for item in sublist])
        complex_array = np.transpose(U2)
        
        # Define a threshold for chopping small values
        threshold = 1e-5

        real_part = np.real(complex_array)
        real_part[np.abs(real_part) < threshold] = 0

        imag_part = np.imag(complex_array)
        imag_part.setflags(write=True)
        imag_part[np.abs(imag_part) < threshold] = 0

        CGCmatrix = real_part + 1j * imag_part
        
        return CGCmatrix


    def _CGCUnitary(self, idx_p, idx_q):
    
        """
        Calculates the CG coefficients for the product of the p  irrep times the q irrep. The ordering (1-based) is based in the character table obtained from GAP.

        Input:
        ----------
        
           - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
           - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
          
          
        Output:
        -------
        
            - CGCmatrixunitary: numpy array. A numpy array
                This numpy array contains the CG coefficients when the group is unitary.
          
        """
        
        
        sp = self.dimensions_of_reps[idx_p-1]
        sq = self.dimensions_of_reps[idx_q-1]
        U = [] #Define the matrix of CG coefficients
        # Define a threshold for chopping small values
        threshold = 1e-5

        #Calculate the CG coefficients from the matrix A()
        for idx_r in range(1, self.number_of_reps + 1):
            lUr = []
            sr = self.dimensions_of_reps[idx_r-1]
            lmulti = self.getMultiplicity(idx_p, idx_q, idx_r)
            if lmulti == 0:
                continue

            lAold = np.array([[self._A(idx_r, idx_p, idx_q, l+1, k+1) for k in range(sr)] for l in range(sr)])
            
            # Chop small numbers
            real_part = np.real(lAold)
            real_part[np.abs(real_part) < threshold] = 0
            imag_part = np.imag(lAold)
            imag_part.setflags(write=True)
            imag_part[np.abs(imag_part) < threshold] = 0
            lAold = real_part + 1j * imag_part
            
            
            # Compute the normalized coefficient lc
            lc = self._getCNormalized(sr, sp, sq, lAold)
            U.append(lc)
            
            for ln in range(2, int(lmulti.real) + 1):
                lAnew = np.array([
                [lAold[ll, lk] - np.dot(np.array([lc[ll]]).T, np.conj(np.array([lc[lk]]))) for lk in range(sr)]
                    for ll in range(sr)])#Aqui
                    
                # Chop small numbers
                real_part = np.real(lAnew)
                real_part[np.abs(real_part) < threshold] = 0
                imag_part = np.imag(lAnew)
                imag_part.setflags(write=True)
                imag_part[np.abs(imag_part) < threshold] = 0
            
                lAnew = real_part + 1j * imag_part
            
                lc = np.array(self._getCNormalized(sr, sp, sq, lAnew))
                
                U.append(lc)
                lAold = lAnew
        U2 = np.array([item for sublist in np.array(U, dtype=object) for item in sublist])
        complex_array = np.transpose(U2)

        

        # Chop small numbers
        real_part = np.real(complex_array)
        real_part[np.abs(real_part) < threshold] = 0
        imag_part = np.imag(complex_array)
        imag_part.setflags(write=True)
        imag_part[np.abs(imag_part) < threshold] = 0

        CGCmatrixunitary = real_part + 1j * imag_part

        return CGCmatrixunitary

    def getCGC(self, idx_p, idx_q):
        """
        Gets the CG coefficient from the CGC attribute. It also checks if a given CG coefficient has been computed before or not to avoid computing the same CG coefficients twice.
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
          
        Output:
        -------
        
            - cg_coeff_array_pq: numpy array
                A numpy array with the CG coefficients of the product p x q.
        
        """
        
        
        if not (1 <= idx_p <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_p = {idx_p}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        if not (1 <= idx_q <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_q = {idx_q}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
            
        
        # Check if the Clebsch-Gordan coefficients have already been calculated
        if self.cg_coeff_calculated[idx_p-1, idx_q-1] != True:
            # Check if the group has unitary representations
            if self.unitary_representations:
                # Calculate the unitary Clebsch-Gordan coefficients
                self.cg_coeff[idx_p-1, idx_q-1] = self._CGCUnitary(idx_p, idx_q)
            else:
                # Calculate the general Clebsch-Gordan coefficients
                self.cg_coeff[idx_p-1, idx_q-1] = self._CGC(idx_p, idx_q)
            
            # Mark the coefficients as calculated
            self.cg_coeff_calculated[idx_p-1, idx_q-1] = True
        
        # Return the Clebsch-Gordan coefficients
        cg_coeff_array_pq = self.cg_coeff[idx_p-1, idx_q-1]
        return cg_coeff_array_pq

    def getCGCcc(self, idx_p, idx_q):
    
        """
        Gets the complex transpose of the matrix with the CG coefficients for the product of irreps p x q. If the group is not unitary, it gets the inverse matrix of the matrix with the CG coefficients. It also checks if this computation has been done before for a given product to avoid computing the same CG coefficients twice.
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
          
        Output:
        -------
        
            - cg_coeff_array_pq_cc: numpy array.
                A numpy array with the complex cojugate matrix (inverse matrix) of the CG coefficients of the product p x q for a unitary (non-unitary) group.
        
        """
        
        if not (1 <= idx_p <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_p = {idx_p}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        if not (1 <= idx_q <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_q = {idx_q}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        
        if self.unitary_representations:
            # Return the conjugate transpose of the Clebsch-Gordan coefficients
            cg_coeff_array_pq_cc = np.conjugate(np.transpose(self.getCGC(idx_p, idx_q)))
            return cg_coeff_array_pq_cc
        else:
            # Return the inverse of the Clebsch-Gordan coefficients
            cg_coeff_array_pq_cc  = np.linalg.inv(self.getCGC(idx_p, idx_q))
            return cg_coeff_array_pq_cc


    def getKroneckerProduct(self, idx_p, idx_q):
    
        """
        Gets the multiplicity of the all irreducible representations obtained in the decomposition of the product of irreps p x q.
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
          
        Output:
        -------
        
             - kronecker_product_array_pq:  numpy array
                A numpy array with the multiplicities of all the irreps in the product of the irreps p x q.
        
        """
        
        
        if not (1 <= idx_p <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_p = {idx_p}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        if not (1 <= idx_q <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_q = {idx_q}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        # Check if the Kronecker product has already been calculated
        if self.kronecker_product_calculated[idx_p-1, idx_q-1] != True:
            # Calculate the Kronecker product using the _generateKroneckerProduct function
            self.kronecker_product[idx_p-1, idx_q-1] = self._generateKroneckerProduct(idx_p, idx_q)
            # Mark the Kronecker product as calculated
            self.kronecker_product_calculated[idx_p-1, idx_q-1] = True
        
        # Return the Kronecker product
        kronecker_product_array_pq= self.kronecker_product[idx_p-1, idx_q-1]
        return kronecker_product_array_pq

    def getMultiplicity(self, idx_p, idx_q, idx_r):
    
        """
        Gets the multiplicity of the the r irrep obtained in the decomposition of the product of irreps p x q .
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_r: int
                Index (1-based) of the r irrep according to character table. The ordering is based in the character table obtained from GAP.
          
        Output:
        -------
        
            - multuplicity_pq: int
                An integer which is the multiplicity of the irrep r in the product of the irreps p x q.
        
        """
        
        
        if not (1 <= idx_p <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_p = {idx_p}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        if not (1 <= idx_q <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_q = {idx_q}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        if not (1 <= idx_r <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_r = {idx_r}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        # Retrieve the Kronecker product for (p, q) and return the r-th element
        multuplicity_pq = round(self.getKroneckerProduct(idx_p, idx_q)[idx_r-1].real)
        return multuplicity_pq

    def _decomposeTensorProduct(self, idx_p, idx_q, vec):
        
        """
        Helper  function that takes a vector  in the product of irreps p x q, and partitions it into sub-vectors according to the multiplicities and dimensions of irreps appearing in the CG decomposition of p  x q.
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - idx_q: int
                Index (1-based) of the q irrep according to character table. The ordering is based in the character table obtained from GAP.
            - vec: numpy array
                Matrix with the result of the tensor product p x q
        
          
        Output:
        -------
        
            - list_irreps: list
                A list with the different irreps of the product p x q obtained. Each entry corresponds to an irrep r in the decomposition p x q.
        
        """
    
        list_irreps = []
        lvec = vec
        
        for idx_r in range(1, self.number_of_reps + 1):
            lmult = self.getMultiplicity(idx_p, idx_q, idx_r)
            ltemp = []
            ldim = self.dimensions_of_reps[idx_r-1]
            
            for li in range(1, lmult + 1):
                lred = lvec[:ldim]
                if not self._VectorZeroQ(lred):
                    ltemp.append(lred)
                lvec = lvec[ldim:]
            
            list_irreps.append(ltemp)
        
        return list_irreps




    def _mJoin(self, list_X, list_Y):
        """
        Helper function for multiply_two. It joins different irreps in a single list of irreps.
        
        Input:
        ----------
        
            - list_X: list of lists of sympy symbols
                The first list to join.
            - list_y: list of lists of sympy symbols
                The second list to join.
           
        Output:
        -------
        
            - list_XY: list
                A list of lists of sympy symbols which contains the concatenation of list_X and list_Y.
        
        """
        
        if not list_X and not list_Y:  # If both are empty, return an empty list
            list_XY = []
            return list_XY
        if not list_X:  # If pX is empty, return pY
            list_XY = list_Y
            return list_XY
        if not list_Y:  # If pY is empty, return pX
            list_XY = list_X
            return list_XY
        list_XY = [px + py for px, py in zip(list_X, list_Y)]
        return list_XY # Otherwise, join the lists



    def getRepVector(self, idx_p, label_p):
    
        """
        Creates a list of lists with sympy symbols using label_p as a label. This list corresponds to a field transforming in the p irrep. The ordering (1-based) is based in the character table obtained from GAP.
        
        Input:
        ----------
        
            - idx_p: int
                Index (1-based) of the p irrep according to character table. The ordering is based in the character table obtained from GAP.
            - label_p: str
                string used to name the fields in this irrep.
        
        Output:
        -------
        
            - repVector: list of sympy symbols
                A list of sympy symbols with a field in a irrep p.
        
        """
        
        
        if not (1 <= idx_p <= len(self.dimensions_of_reps)):
            raise ValueError(
                f"Invalid irrep index: idx_p = {idx_p}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
                        
        repVector = [[] for _ in range(self.number_of_reps)]  # Retrieve the representation vector
        pRep = int(idx_p)
        if pRep > 0:
            repVector[pRep - 1] = [self._genLabels(self.dimensions_of_reps[pRep-1], label_p)]
        
        return repVector


    def _genLabels(self, dim_p, label_p):
    
        """
        Creates the different labels for the getRepVector method.
        
        Input:
        ----------
        
        
           - dim_p: int
               the dimension of the irrep
           - label_p: str
               string used to name the fields in this irrep. For instance, if label_p = "x", then they will be named as sp.Symbolx("xr1"), sp.Symbolx("xr2"), ...
        
        Output:
        -------
        
            - labels: list of sympy symbols
                A list of sympy symbols to generate the field that transforms in a given irrep.
        
        """
        labels = []
        for lf in range(1, dim_p + 1):
            var_name = f"{label_p}r{lf}"
            symbol = sp.Symbol(var_name)
            labels.append(symbol)
        return labels

    def _multOneRep(self, idx_li, idx_lj, list_A, list_B):
        
        """
        Calculates the product of the index idx_li irrep of the list list_A with the idx_lj-th irrep of the list list_B. Here list_A and list_B can be fields that only transform in one irrep. Or they can be lists with different irreps.
        
        Input:
        ----------
        
            -  idx_li: int
               Index (1-based) of the li irrep of the list_A. The ordering is based in the character table obtained from GAP.
            - idx_lj: int
               Index (1-based) of the li irrep of the list_A. The ordering is based in the character table obtained from GAP.
            - list_A : list of sympy symbols
               A list that contains different irreps.
            - list_B : list of sympy symbols
               A list that contains different irreps.
        
        Output:
        -------
        
            - mult: list of sympy symbols
                A list of sympy symbols. This list contains the product of the li irrep of list_A with the lj irrep of list_B.
        
        """

        def process_y(y): #Helper functions to compute the tensor product between two irreps
            def process_x(x):
                tensor_product = np.kron(x, y) # Aqui
                transformed_tensor = np.dot(self.getCGCcc(idx_li, idx_lj), tensor_product)
                return self._decomposeTensorProduct(idx_li, idx_lj, transformed_tensor)
            
            return list(map(process_x, list_A[idx_li-1]))  # Adjusted for 0-based indexing
        
        mult = list(map(process_y, list_B[idx_lj-1]))
        return mult # Adjusted for 0-based indexing


    def _eliminate_small_real_imag(self, expr):
        """
        Helper function to eliminate small real and imaginary parts in sympy expressions
        
        Input:
        ----------
    
        - expr: sympy Symbol
        

        Output:
        -------
            - expr_chopped: sympy symbol.
                A sympy Symbol with chopped real and imaginary numbers.
        
        """
        def replacement(f): #Helper function to chop a float
            real_part = f.as_real_imag()[0]
            imag_part = f.as_real_imag()[1]

            new_real = 0 if abs(real_part) < 1e-5 else real_part
            new_imag = 0 if abs(imag_part) < 1e-5 else imag_part
            
            return new_real + sp.I*new_imag
        
        expr_chopped = expr.replace(lambda e: e.is_Float, replacement)
        return expr_chopped



    
    def _flatten_once(self, lst):
        """
        Helper function that acts as Mathematica's Flatten level 1.
        
        Input:
        ---------
        
            - lst: A list
                A list which each element is a list
           
           
        Output:
        -------
        
            - flatted_lst: list.
                A list where all the lists inside lst were concatanated.
        """
        flatted_lst = list(itertools.chain.from_iterable(lst))
        return flatted_lst

    def _multiply_two(self, list_A, list_B):
    
        """
        It gives a list with the different irreps obtained from the tensor product between the vector list_A with list_B.
        
        
        Input:
        ----------
        
            - list_A: list of lists with sympy symbols
                This is the first list with irreps to use in the tensor multiplication.
            - list_B: list of lists with sympy symbols
                This is the second list with irreps to use in the tensor multiplication.
           
           
        Output:
        -------
        
            - product:A list of lists with sympy symbols.
                This list contains all the irreps resulting from the product of list_A x list_B.
        
        """
        lres = [[] for _ in range(self.number_of_reps)]
        
        if not isinstance(list_A, list):
            raise ValueError("Expected a list for 'list_A', but got a non-list object.")

        if len(lres) != len(list_A):
            raise ValueError(f"Length mismatch: expected {len(lres)} elements, got {len(list_A)}.")

        if not all(isinstance(sublist, list) for sublist in list_A):
            raise ValueError("Each element of 'list_A' must be a list.")
            
        if not isinstance(list_B, list):
            raise ValueError("Expected a list for 'list_B', but got a non-list object.")

        if len(lres) != len(list_B):
            raise ValueError(f"Length mismatch: expected {len(lres)} elements, got {len(list_B)}.")

        if not all(isinstance(sublist, list) for sublist in list_B):
            raise ValueError("Each element of 'list_B' must be a list.")
            
        
        
        if list_A == lres:
            product = list_B
            return product
        
        if list_B == lres:
            product = list_A
            return product
        
        for li in range(len(list_A)):
            for lj in range(len(list_B)):
                if li > lj:
                    ltemp = self._multOneRep(lj + 1, li + 1, list_B, list_A)
                else:
                    ltemp = self._multOneRep(li + 1, lj + 1, list_A, list_B)
                
                flattened_temp = self._flatten_once(ltemp)# Flatten by one level
                if not not flattened_temp:
                    for k in range(len(flattened_temp)):
                           lres = self._mJoin(flattened_temp[k],lres)
            

        result = [[[self._eliminate_small_real_imag(expr) for expr in subsublist] for subsublist in sublist] for sublist in lres]
        
        # Create a set to track unique expressions
        product  = []
        for i in range(len(result)):
            unique_exprs = []
            for expr in result[i]:
                if not any(expr[0] - unique[0] == 0 for unique in unique_exprs):
                    unique_exprs.append(expr)
            product.append(unique_exprs)
        

        return product
        
    def multiply(self,list_fields):
        """
        It gives a list with the different irreps obtained from the tensor product of the fields included in the input.
        
        Inputs:
        -------
        
            - list_fields: list of lists with sympy symbols
                This list include all the fields to be multiplied
                  
        Outputs:
        --------
        
        
             - res: list of lists with sympy symbols
                Irrep decomposition of the product of the fields in list_fields
             
        """
    
        res = list_fields[0]
        for i in range(1, len(list_fields)):
            res = self._multiply_two(res, list_fields[i])
        return res
        


    def extractOneRep(self, vec_rep, idx_p):
        """
        It extracts one the index (1-base) p irrep from the list of irreps vec_rep. The ordering of the irreps (1-based) is based in the character table obtained from GAP.
        
        Input:
        ----------
        
             - vec_rep: list of lists of symbols
                This is the list of irreps from which you will extract only one of the irreps.
             - idx_p:  int
                This is the index (1-based) of the p irrep that you want to extract.
            
        Output:
        -------
        
              - oneRep_list: A list of lists of symbols with the p irrep of the vector vec_rep.
                This list will contain mostly empty lists except from the irrep you extracted.
           
        """
        
        lres = [[] for _ in range(self.number_of_reps)]
        
        
        if not isinstance(vec_rep, list):
            raise ValueError("Expected a list for 'vec_rep', but got a non-list object.")

        if len(lres) != len(vec_rep):
            raise ValueError(f"Length mismatch: expected {len(lres)} elements, got {len(vec_rep)}.")

        if not all(isinstance(sublist, list) for sublist in vec_rep):
            raise ValueError("Each element of 'vec_rep' must be a list.")
            
            
        if not (1 <= idx_p <= len(self.dimensions_of_reps)):
            raise ValueError(
            f"Invalid irrep index: idx_p = {idx_p}. It must be between 1 and {len(self.dimensions_of_reps)} (inclusive)."
            )
        
        idx_ppython = idx_p - 1
        oneRep_list = [elem if idx == idx_ppython else [] for idx, elem in enumerate(vec_rep)]
        return oneRep_list


    def _flatten(self, nested_list):
        """
        It flattens like mathematica.
        
        Input:
        ----------
        
             - nested_list: list of lists
                This list will be flattened like Mathematica
            
        Output:
        -------
        
              - flat_list: A list.
                A flattened list
        """
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(self._flatten(item))
            else:
                flat_list.append(item)
        return flat_list
