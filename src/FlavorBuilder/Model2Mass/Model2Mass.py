import numpy as np
import sympy as sp
import json
import os
from itertools import product
from itertools import permutations
from FlavorBuilder.PyDiscrete import Group
import random
from sympy import symbols, expand, Symbol
from collections import defaultdict
from collections import Counter
from numbers import Real
import re







class Model2Mass():

    def __init__(self, group, model_file = None,  model = None, vev_dict = None, non_flavon_particles = 11, particle_list_grouping = None):
    
    
        """
        
        Inputs:
        --------
            - group: PyDiscrete group object.
                A group obtained by the PyDiscrete package.
                
            - model_file: str
                A string containing the path to the file containing the model.
                
            - model: dictionary
                A dictionary containing all the necessary details to extract the lagrangian and mass matrices. See the tutorial for details.
                
            - vev_dict: dictionary (Optional. The default is set by self._setFlavonVeVsdict(). See the docstring therein)
                A dictionary that will specify the vevs for the different vevs.
                
            - non_flavon_particles: int (Optional. Set to 11 by default)
                An integer specifying the number of non-flavon particles.
                
            - particle_list_grouping: dictionary (Optional. The default is set by _setParticleListGrouping(). See the docstring therein)
                A list of lists  with two strings, specifying the left leptons, right leptons, ... and so on.
                
                
                
                
        Atrributes:
        ----------
            - model_file: str
                Path to the model_file
            - model: dictionary
                Model dictionary with all the necessary specifications to build a lagrangian. See the tutorial for more details.
            - abelian_group_dimensions: array
                List with the dimensions of the ZN symmetries. It is obtained from the "ZN_symmetries" key of the model dictionary.
            - parametersSet: dictionary
                Dictionary with the values that were set in the model. For instance, the flavon vevs, or the vevs of the higgs to ignore them when calculating the dimensionless mass matrix.
            - flavonvevs: list of sympy symbols
                Flavon sympy symbols
            - list_of_fields_sympy_symbols: list of sympy symbols
                Sympy symbols for all the fields considered
            - list_of_ALL_parameters_in_lag: list of sympy symbols
                Sympy symbols for all the paramaeters in the lagrangian (denoted by alpha{n})
            - numberofAbelianSymmetries: int
                Number of Abelian Symmetries
            - NumberOfTotalFlavons: int
                Number of flavons with a non-zero array in the "representations" key of the model dictionary
            - particle_dict: dictionary
                Dictinary with all the particles in a notation digestible for PyDiscrete
            - non_flavon_particles: int
                Number of particles that are non-flavon

        Methods:
        -----------
        
            - fields_for_discrete(data_model):
                Produce matrix of field representations in format ingestible by PyDiscrete.
                

                
            - get_lagrangian(max_terms = 50, allOperators = None):
                Get the lagrangian and a list of parameters with its respective term in the lagrangian. The max_terms sets the number of terms in the lagrangian after which the computations stops. allOperators is a list of lists with strings that indicates which terms should be compueted for the lagrangian. See the tutorial for more details.
            
            - ParametersInAMatrix(mat):
                Gets the parameters in a sympy matrix mat
            
            - get_rank_mass_matrix(mass_matrix, parameters):
                Gets the rank of mass_matrix. A list of parameters obtained from get_lagrangian must be provided
            
            - count_eff_flavons(lagrangian):
                Gives the number of flavons that appear in the lagrangian and a list with the name of the flavons that do not appear in the lagrangian
            
            - get_mass_matrix(lagrangian ,parameters, left_fields, right_fields = None, higgs_fields = None,  factor_out = True):
                Gets the mass matrix of the lagrangian. It needs the parameters output from get_lagrangian. Also, it needs a list of left_fields, right_fields and higgs_fields to perform the derivatives to calculate the mass matrix. The factor_out input, if true, factors out one of the parameters and sets it to one in the model. This is useful since it makes the fitting process faster.
        
        
        """
       
       
        if group == None:
           raise ValueError("You must provide a group from PyDiscrete")
        self.group = group
        
        if (model_file is None) and (model is None):
            raise ValueError("You must provide a model file or a model.")
     
        self.model_file = model_file
        self.model = model
        
        
        self.abelian_group_dimensions = 0
        self.parametersSet = {}
        self.flavonvevs = []
        self.list_of_fields_sympy_symbols = []
        self.list_of_ALL_parameters_in_lag = []
        self.numberofAbelianSymmetries = 0
        self.NumberOfTotalFlavons = 0
        self.particle_dict = {}
        
        self.non_flavon_particles = non_flavon_particles
        
        
        
        self.model = self._read_model()
        self._setParticleListGrouping(particle_list_grouping)
        self._setFlavonVeVsdict(vev_dict)
        


        

    def _read_model(self):
        '''
        Format data for PyDiscrete. It saves the information of the model_file into python variables to be handled easily

        Output:
        -------
            - data_model: dictionary
                This dictionary contains the particle representation information obtained from the model_file with absent particles removed.
        '''
        if self.model_file is not None:
            with open(self.model_file, 'r') as file:
                data_model0 = json.load(file)
        if self.model is not None:
            data_model0 = self.model
        if (self.model is not None) and (self.model_file is not None):
            with open(self.model_file, 'r') as file:
                tempdata_model0 = json.load(file)
            if self.model != tempdata_model0:
                raise  ValueError("You gave both a model and a model file but they are not equal. They should be equal.")
            
        # Extract particles, representations, and associations
        particles = data_model0["particles"]
        representations = data_model0["representations"]
        fixed_flavon_vevs = data_model0["fixed_flavon_vevs"]
        ZN_symmetries = data_model0["ZN_symmetries"]
        self.numberofAbelianSymmetries = len(representations[0]) - self.group.number_of_reps
        self.abelian_group_dimensions = ZN_symmetries
        self.parametersSet = {}
             
        # Remove all zero entries
        data_model = {
            'representations': [],
            'associations': [],
            'particles': [],
            'fixed_flavon_vevs': [],
            'ZN_symmetries': []
        }
        rep_array = np.array(representations)
        all_zero_rows = np.all(rep_array == 0, axis=1)
        mask = np.ones(rep_array.shape[0], dtype=bool)
        mask[all_zero_rows] = False
        data_model['representations'] = rep_array[mask].tolist()
        data_model['associations'] = data_model0['associations']
        data_model['particles'] = np.array(particles)[mask].tolist()
        data_model['ZN_symmetries'] = ZN_symmetries

                
        rep_array_vevs = np.array(fixed_flavon_vevs)
        self.NumberOfTotalFlavons = len(rep_array_vevs)
        all_zero_rows_flavonsVevs = all_zero_rows[self.non_flavon_particles:]
        mask_flavonsVevs = np.ones(np.array(rep_array_vevs).shape[0], dtype=bool)
        mask_flavonsVevs[all_zero_rows_flavonsVevs] = False
        
        
        data_model['fixed_flavon_vevs'] = rep_array_vevs[mask_flavonsVevs].tolist()
        

        return data_model

    def fields_for_discrete(self, data_model):
        '''
        Produce matrix of field representations in format ingestible
        by PyDiscrete.

        Inputs:
        -------
        
            - data_model: Dictionary
                 This is a dictionary of particle representation information.

        Outputs:
        --------
        
            - fieldsForDiscrete: A matrix of field representations in
                                 PyDiscrete friendly format.
        '''
        
        
        SeesawForDiscrete_lst = self._SeesawForDiscrete(data_model) #This prepares the MSSM + 3 right-handed neutrinos for PyDiscrete
        flavonsForDiscrete_lst = self._read_flavons(data_model) #This prepares the flavons for PyDiscrete
        
        
        fieldsForDiscrete = SeesawForDiscrete_lst + flavonsForDiscrete_lst #This list will include all MSSM fields + 3 right-handed neutrinos + flavons

        return fieldsForDiscrete
        
    def _SeesawNpletsWithRow(self, data_model):
        """
        Helper function to group the MSSM and the 3 right-handed neutrinos appropiately. For instance, if the three left SU(2) doublets belong to a triplet, they
        need to be grouped together.
        
        Inputs:
        -------
        
            - data_model: Dictionary
                 This is a dictionary of particle representation information.
                 
        Outputs:
            - SeesawNpletsWithRow: list
                 This is a list that encodes the information of the reps for the MSSM + 3 right-handed neutrinos. It also groups together particles that belong to the same multiplet.
        
        """
        
        
        # List to store the results
        SeesawNpletsWithRow = []
        already_grouped = set()


        # Loop through the first eleven particles which correspond to the MSSM + 3 right-handed neutrinos
        for i, particle in enumerate(data_model['particles'][:self.non_flavon_particles]):
            
            # Skip if the particle is already grouped
            if particle in already_grouped:
                continue
            
            representation = data_model['representations'][i]
            

            rep_num = np.where(representation)[0][0] + 1
            
            # Check if the irrep is a singlet
            if rep_num <= self.group.NumberOfSinglets:
            
                # Add single particle entry (particle, A4 rep, ZN rep)
                SeesawNpletsWithRow.append([[particle], rep_num, representation[self.group.number_of_reps:self.group.number_of_reps+self.numberofAbelianSymmetries]])
                already_grouped.add(particle)
            else:
                # Group particles with the same non-zero entry
                group = [particle]
                already_grouped.add(particle)
                in_group = np.where(np.array(data_model['associations']) == data_model['associations'][i])[0][1:]
                for k in in_group:
                    other_particle = data_model['particles'][k]
                    other_representation = data_model['representations'][k]
                    group.append(other_particle)
                    already_grouped.add(other_particle)

                # Add the field to the list SeesawNpletsWithRow depending on whether they are singlets or not
                if len(group) > 1: #Non-singlets
                    SeesawNpletsWithRow.append([group, rep_num, representation[self.group.number_of_reps:self.group.number_of_reps+self.numberofAbelianSymmetries]])
                else: #Singlet
                    SeesawNpletsWithRow.append([[particle], rep_num, representation[self.group.number_of_reps:self.group.number_of_reps+self.numberofAbelianSymmetries]])
                    
        return SeesawNpletsWithRow
        
        
    def _SeesawForDiscrete(self, data):
        """
        Helper function to make  the MSSM and the 3 right-handed neutrinos digestible for PyDiscrete.
        
        Inputs:
        -------
        
            - data_model: Dictionary
                 This is a dictionary of particle representation information.
                 
        Outputs:
            - MSSM3NForDiscrete: list
                 This is a list that encodes the information of the reps for the MSSM + 3 right-handed neutrinos digestible for PyDiscrete format.
        
        """
         
        SeesawNpletsWithRow = self._SeesawNpletsWithRow(data)
        MSSM3NForDiscrete = []
        for nplet in SeesawNpletsWithRow: #Add MSSM fields + 3 right-handed neutrinos to fieldsForDiscrete
            repNumber = int(nplet[1])
            if repNumber <= self.group.NumberOfSinglets:
                for j in range(len(nplet[0])):
                    fieldname = nplet[0][j]
                    var_name = f"{fieldname}"
                    symbol = sp.Symbol(var_name)
                    self.list_of_fields_sympy_symbols.append(symbol)
                    lres = [[] for i in range(self.group.number_of_reps)]  # Retrieve the representation vector
                    lres[repNumber - 1] = [[symbol]]
                    MSSM3NForDiscrete.append([lres,nplet[-1:][0]])
            else: #Create n-plets for flavons and add them to fieldsForDiscrete
                lres = [[] for i in range(self.group.number_of_reps)]  # Retrieve the representation vector
                lres2 =[]
                for j in range(len(nplet[0])):
                    fieldname = nplet[0][j]
                    var_name = f"{fieldname}"
                    symbol = sp.Symbol(var_name)
                    self.list_of_fields_sympy_symbols.append(symbol)
                    lres2.append(symbol)
                lres[repNumber - 1] = [lres2]
                MSSM3NForDiscrete.append([lres,nplet[-1:][0]])
                
        return MSSM3NForDiscrete
        
        
    def _setFlavonVeVsdict(self, vev_dict):
        """
        This function sets the identification of the different flavon vevs you consider
        
        Inputs:
        ------
        
            - vev_dict: Dictionary
                This is a dictionary between integer numbers and flavon vevs. If none, it goes to the defaul list given by:
                {1 :  np.array([1]), 2: np.array([1, 0, 0]), 3: np.array([1, 1, 1]), 4: np.array([0, 1, 0]), 5: np.array([0, 0, 1]), 6: np.array([0, 1, -1]) , 7 : np.array([1, -(-1)**(1/3), (-1)**(2/3)]) , 8: np.array([1, (-1)**(2/3), -(-1)**(1/3)]) }
                
        Outputs:
        --------
        
        None
        
        """
        if vev_dict == None:
            self.vev_dict =  {1 :  np.array([1]), 2: np.array([1, 0, 0]), 3: np.array([1, 1, 1]), 4: np.array([0, 1, 0]), 5: np.array([0, 0, 1]), 6: np.array([0, 1, -1]) , 7 : np.array([1, -(-1)**(1/3), (-1)**(2/3)]) , 8: np.array([1, (-1)**(2/3), -(-1)**(1/3)]) }
        else:
            self.vec_dict = vev_dict
            
        if not all(isinstance(key, int) for key in self.vev_dict.keys()):
            raise ValueError("All the keys must be integers.")
        if not all( (isinstance(val, np.ndarray) and (np.issubdtype(val.dtype, np.floating) or np.issubdtype(val.dtype, np.integer) or np.issubdtype(val.dtype, np.complexfloating))) or (isinstance(val, list) and all(isinstance(x, (float, int, complex)) for x in val)) for val in self.vev_dict.values() ):
            raise ValueError("All values must be lists or NumPy arrays containing only floats")
            
            
            
    def _setParticleListGrouping(self, particle_list_grouping):
        
        """
        
        It sets the attribute particle_list_grouping in the model2mass class.
        Inputs:
        -------
            - particle_list_grouping: list of strings
                 This is a list of particles that will be used to group them to contruct the lagrangian terms. If particle_list_grouping = None, it gives the default [["leftLeptons", "L"], ["rightLeptons", "E"], ["rightNeutrinos", "N"],  ["flavons", "phi"], ["higgsup", "Hu"], ["higgsdown", "Hd"]]

        """
        
        
        if particle_list_grouping == None:
            self.particle_list_grouping = [["leftLeptons", "L"], ["rightLeptons", "E"], ["rightNeutrinos", "N"],  ["flavons", "phi"], ["higgsup", "Hu"], ["higgsdown", "Hd"]]
        
        else:
            if not all( isinstance(elem, list) and len(elem) == 2 and all(isinstance(sub_elem, str) for sub_elem in elem) for elem in particle_list_grouping ):
                raise ValueError("All elements of particle_list_grouping must be lists of exactly two strings.")
                
            self.particle_list_grouping = particle_list_grouping
        
        
             

    def _assignFixedVevs(self, fixed_flavon_vevs_index, repPosition, symbolr ):
        """
            This is a helper function that saves the fixed vevs selected by the user into the attribute ""parametersSet"". This attributed is later used when evaluating the mass matrices
            
        Inputs:
        -------
            - fixed_flavon_vevs_index: int
                This is the index that determines from the attribute "vev_dict" which vev will be assigned to the flavon
            - repPosition: int
                This is the index of the irrep for the flavon to be assigned a vev. It should follow the conventions by PyDiscrete.
            - symbolr : sympy symbol
                This is the sympy symbol that will be used to assign the name of this vev.
        """
        flavonInPydiscrete = self.group.getRepVector(repPosition, symbolr)
        dimOfRep = self.group.dimensions_of_reps[repPosition - 1] # the -1 is to follow the 0-index notation in python
        dimOfVeV = len(self.vev_dict[fixed_flavon_vevs_index])
        
        if  not dimOfVeV == dimOfRep:
            raise ValueError("The dimension that you selected for the flavon does not match the assigned vev in the dictionary you choose")
        
        for i in range(dimOfRep):
            self.parametersSet[self.group._flatten(self.group.getRepVector(repPosition, symbolr))[i]] = self.vev_dict[fixed_flavon_vevs_index][i]
        
        return None
        
        
    def _read_flavons(self, data_model):
        """
        Helper function to make flavons be part of a list digestible for PyDiscrete. It also sets the flavon vevs and saves their value in the class attribute parametersSet.
        
        Inputs:
        -------
        
            - data_model: Dictionary
                 This is a dictionary of particle representation information.
                 
        Outputs:
            - flavonsForDiscrete: list
                 This is a list that encodes the information of the reps for flavons digestible for PyDiscrete format.
        
        """
        flavonsForDiscrete = []
        
        flavonsinput = []
        for i in range(self.non_flavon_particles, len(data_model['particles'])):
            flavonsinput.append([data_model['particles'][i], data_model['representations'][i]])

        flavonvevs = []


        # Define the function to process each row
        for i in range(len(flavonsinput)):
            row = flavonsinput[i]
            symbol = row[0]
            symbolr = symbol + 'r'
            values = row[1]
            repPosition = np.where(values)[0][0] + 1
            self.list_of_fields_sympy_symbols.append(sp.Symbol(symbolr))

            # Construct the code string
            flavonsForDiscrete.append([self.group.getRepVector(repPosition, symbolr), row[1][self.group.number_of_reps:self.group.number_of_reps+self.numberofAbelianSymmetries]])
            flavonvevs.append(self.group.getRepVector(repPosition, symbolr))
            
            
            #Call _assignFixedVevs to assign vecs from vev_dict for later use
            fixed_flavon_vevs_index= data_model['fixed_flavon_vevs'][i]
            self._assignFixedVevs(fixed_flavon_vevs_index, repPosition, symbolr )
            
        
        self.flavonvevs = self.group._flatten(flavonvevs)
        
            
        return flavonsForDiscrete

    # Function to scan symbols inside each row
    def _scanRow(self, row, particle_dict, particle_list_grouping):
        '''
        Identify particles present in each row and assign to correct list in particle dictionary. It will assign the particle to the correct entry in the particle dictionary. This entry can be "left leptons" for the left SU(2) doublets leptons, "rightLeptons" for right SU(2) singlet leptons, "rightNeutrinos" for right-handed neutrinos, "higgsup" for Up higgs  , "higgsdown" for the down higgs and "flavons" for flavons.

        Inputs:
            - row: list.
                A list of representations for a single particle.
            - particle_dict: dictionary
                Dictionary categorizing particles.

        Outputs:
            - particle_dict: dictionaty
                Updated with information from row.
        '''
        
        
        
        # Flatten the row
        flattenedRow = [item for sublist in row for item in sublist]
        # Convert to strings (if not already)
        symbols = [str(item) for item in flattenedRow]

        
        # Append to the appropriate list based on the symbol content
        for i in range(len(particle_list_grouping)):
            if any(particle_list_grouping[i][1] in symbol for symbol in symbols):
                particle_dict[particle_list_grouping[i][0]].append(row)
        
        return particle_dict

    def _get_particle_dict(self, fieldsForDiscrete):
        '''
        Construct particle dictionary from fields. By default, it groups all left SU(2) doublets leptons, right SU(2) singlet leptons,
        right-handed neutrinos, higgs up , higgs down and flavons in a particle dictionary. It can also be made customary

        Input:
            - fieldsForDiscrete: list of sympy symbols
                 A list of sympy field representations in PyDiscrete friendly format.
        Output:
            - particle_dict: dictionary
                 Dictionary categorizing particles with their representations.
        '''

        
        particle_dict = {key[0]: [] for key in self.particle_list_grouping}
        
        # Scan each row in the fields
        for row in fieldsForDiscrete:
            particle_dict = self._scanRow(row, particle_dict, self.particle_list_grouping)
        return particle_dict
  
    
    # Function to return unique indices and indices of repeated lists
    def _sort_symbols(self, original_list):
        """
        Helper function that sorts symbols from a list
        
        Inputs:
        -------
            - original_list: list of sympy symbols
        
        Outputs:
        --------
            - sorted_list: sorted list of sympy symbols
        
        """
        unique_elements = []
        repeated_elements_dict = {}
    
        for idx, element in enumerate(original_list):
            if original_list.count(element) == 1:
                unique_elements.append(idx)
            else:
                if str(element) not in repeated_elements_dict:
                    repeated_elements_dict[str(element)] = [idx]
                else:
                    repeated_elements_dict[str(element)].append(idx)
    
        result = [unique_elements]
        for indices in repeated_elements_dict.values():
            result.append(indices)
        
        sorted_list = []
        for i in range(len(result)):
            temp_list = [original_list[j] for j in result[i]]
            sorted_list = sorted_list + temp_list
    
        return sorted_list

                                
    def _unique_combs_fields(self, list_fields, repeated_fields):
        '''
        This is a helper function that helps the multiplication in the lagrangian to not make repetitions.
        
        Inputs:
        -------
            - list_fields: list of sympy symbols:
                This list is the list of fields for each category.
            
            
            - repeated_fields: list of integers
                This list a list of integers where all the elements with the same integer will be grouped together since this means they are the same field.
            
        Outputs:
        --------
            - valid_tuples: list of tuples
                This list includes the valid indices such that the products in the lagrangian do not repeat.
        
        '''
        
        # Build index ranges for each list
        index_ranges = [range(len(f)) for f in list_fields]

        # Identify groups from repeated_fields
        group_to_indices = defaultdict(list)
        for idx, tag in enumerate(repeated_fields):
            if tag != 0:
                group_to_indices[tag].append(idx)

        # Generate full list of tuples
        all_index_tuples = product(*index_ranges)



        #Filter combinations that satisfy group-wise monotonicity
        valid_tuples = []
        for idx_tuple in all_index_tuples:
            is_valid = True
            for group_indices in group_to_indices.values():
                group_values = [idx_tuple[i] for i in group_indices]
                if group_values != sorted(group_values):
                    is_valid = False
                    break
            if is_valid:
                valid_tuples.append(idx_tuple)

        
        return valid_tuples
        
    def _operatorToLag(self, list_fields, lagrangian, parameters, n_terms, coeff, max_terms, repeated_fields):
    
        '''
        Construct lagrangian and parameters for the NN operator

        Input:
        -----
            - list_fields: list of lists of sympy symbols
                 Each element is a type of particle (flavons, right-handed neutrinos) category and within each category there are the irreps of those fields.
            - lagrangian: expression of sympy symbols
                 The lagrangian to which we will add the result of the operator "NN".
            - parameters: list tuples.
                 Each tuple is of the form (alpha_k, coeff, term_k) where alpha_k is the coefficient of the term in the lagrangian. Thus, the lagrangian looks like lag = \sum_k \alpha_k * coeff * term_k. Coeff encodes how much supressed is this term due to higher-order powers of \phi/\Lambda.
            - n_terms: int
                 Number of terms computed so far for the lagrangian.
            - max_terms: int
                 Maximum number of terms allowed in the lagrangian.
            - k: int
                 The index of the term at which the computation of the invariant term  starts.
            - repeated_fields: list of indices
                 This is a list that indicates which of the elements in list_fields are repeated

        Output:
        -------
            - lagrangian: expression of sympy symbols.
                  The input lagrangian  + the terms from the operator obtained from multiplying the fields in list_fields
            - parameters: list tuples.
                  The parameters input list with the new terms appended.
            - n_terms: int
                  The input n_terms + the number of terms computed in this function

                  
                  
                  
        '''
    
        valid_tuples = self._unique_combs_fields(list_fields, repeated_fields)
        
        for idx_tuple in valid_tuples:
            if n_terms > max_terms:
                break
            mod_condition = [int(sum((list_fields[nn][idx_tuple[nn]][1][jjj] - 1) for nn in range(len(list_fields)))) % self.abelian_group_dimensions[jjj] == 0 for jjj in range(self.numberofAbelianSymmetries)]

            if all(mod_condition):
                # Get invariant terms by calling the group multiplication from PyDiscrete

                sorted_terms = self._sort_symbols([list_fields[nn][idx_tuple[nn]] for nn in range (len(list_fields))])
                invariant_terms = self.group._flatten(self.group.extractOneRep(self.group.multiply([sorted_terms[nn][0] for nn in range(len(sorted_terms))]), 1))
                                    
                
                if invariant_terms:  # Check if invariant_terms is not empty
                    for nnn in range(len(invariant_terms)):
                        alpha_k = sp.symbols(f'alpha{n_terms + 1}')
                        var_name = f'alpha{n_terms + 1}'
                        symbol = sp.Symbol(var_name)
                        lagrangian += coeff * alpha_k * invariant_terms[nnn]
                        self.list_of_ALL_parameters_in_lag.append(alpha_k)
                        parameters.append([alpha_k, coeff, invariant_terms[nnn], [sorted_terms[nn] for nn in range(len(sorted_terms))]])
                        n_terms += 1
        return lagrangian, parameters, n_terms



    def _label_repeated_fields(self, list_fields):
        """
        This is a helper function that gives the correct list of integers for repeeated fields. It is used in get_lagrangian.
        
        Inputs:
        -------
                - list_fields: list of strings
                     List with the fields to obtain the repeated list
        
        Outputs:
        --------
                - list_repeated_fields: list of integers
                      List that indicates which fields are repeated.
        
        
        """
        counts = Counter(list_fields)       # Count each string's occurrences
        list_repeated_fields = [None] * len(list_fields)   # Output list
    
        repeat_label = 1                 # Start numbering repeated types from 1
        repeat_map = {}                  # Map repeated string → assigned label
    
        for i, s in enumerate(list_fields):
            if counts[s] == 1:
                list_repeated_fields[i] = 0  # Unique string
            else:
                if s not in repeat_map:
                    repeat_map[s] = repeat_label
                    repeat_label += 1
                list_repeated_fields[i] = repeat_map[s]
    
        return list_repeated_fields
    
    def _setallOperators(self, allOperators = None):
        """
        This is a helper function which saves the list of all operators that the user wants to compute
        
        Inputs:
        -------
              - allOperators: list of list with strings and coefficients
                   This is the list that contains all the terms you want to be considered in the Lagrngian. The first element of each entry should be a list with strings and the second a float which sets the supression coefficient. If None, it sets to the default:
                               [
                        [["rightNeutrinos", "rightNeutrinos"], 1],
                        [["flavons", "rightNeutrinos", "rightNeutrinos"], 0.1],
                        [["flavons", "flavons", "rightNeutrinos", "rightNeutrinos"], 0.01],
                        [["rightNeutrinos", "leftLeptons", "higgsup"], 1],
                        [["rightNeutrinos", "leftLeptons", "higgsup", "flavons"], 0.1],
                        [["leftLeptons", "rightLeptons", "higgsdown"], 1],
                        [["leftLeptons", "rightLeptons", "higgsdown", "flavons"], 0.1]
                    ]
                   
        Outputs:
        --------
              - None
        
        """
        
        if allOperators == None:
            self.allOperators = [
                        [["rightNeutrinos", "rightNeutrinos"], 1],
                        [["flavons", "rightNeutrinos", "rightNeutrinos"], 0.1],
                        [["flavons", "flavons", "rightNeutrinos", "rightNeutrinos"], 0.01],
                        [["rightNeutrinos", "leftLeptons", "higgsup"], 1],
                        [["rightNeutrinos", "leftLeptons", "higgsup", "flavons"], 0.1],
                        [["leftLeptons", "rightLeptons", "higgsdown"], 1],
                        [["leftLeptons", "rightLeptons", "higgsdown", "flavons"], 0.1]
                    ]
        else:
            self.allOperators = allOperators
        
        if not all(isinstance(elem, list) for elem in self.allOperators):
            raise ValueError("The first item of each element should be a list.")
        if not all(  all( isinstance(s, str)  for s in elem[0]) and   isinstance(elem[1], Real)    for elem in self.allOperators):
            raise ValueError("The first item of each element should be a list of strings and the second item a real number.")
        
    
    def get_lagrangian(self, max_terms = 50, allOperators = None):
        '''
        Construct the Lagrangian from a table of particle representations.

        Inputs:
        -------
        
        
            - model: int.
                This is the maximm number of terms allowed to have for the lagrangian. The lagrangian computation stops after this number is reached
        Outputs:
        ---------
        
            - lagrangian: sympy expression
                Sympy expression of Lagrangian.
            - parameters: list of sympy symbols
                List of symbols used in Lagrangian. Each element has four elements. The first element corresponds to the coefficient for that term. Note that this coefficient could have been set to one in the matrix multiplication since we can always factor out a coefficient in the mass matrix without loss of generality. The second element is the coefficient (usually of the form 10^{n} for some integer n) that denotes the supression from higher-order term. The third element is the sympy expressin of the lagrangian before evaluating any flavon vevs. The last entry is a list of the fields involve in this term writen as [PyDiscreteField, [Non-Abelian reps] ].
        '''
        fieldsForDiscrete = self.fields_for_discrete(self.model)
        self.particle_dict = self._get_particle_dict(fieldsForDiscrete)
        self._setallOperators(allOperators)


        
        lagrangian = 0
        parameters = []
        n_terms = 0
        
        # Loop through allPperators
        for i in range(len(self.allOperators)):
            if n_terms > max_terms:
                 break
            repeated_fields = self._label_repeated_fields(self.allOperators[i][0])
            list_fields = [self.particle_dict[s] for s in self.allOperators[i][0]]
            coeff = self.allOperators[i][1]
            lagrangian, parameters, n_terms = self._operatorToLag(list_fields, lagrangian, parameters, n_terms, coeff, max_terms, repeated_fields)


        return lagrangian, parameters

    
        

    def ParametersInAMatrix(self, mat):
        '''
        List the symbols of the parameters used in a matrix.

        Input:
            - mat: A sympy matrix.
                The matrix from which we will extract the parameter symbols.
        Output:
            - params_in_mat: List of symbols in the matrix.
                This are the sympy symbols which are the parameters in the matrix.
        '''
        # Extract all symbols in the matrix
        all_symbols = set()
        for element in mat:
            all_symbols.update(element.atoms(sp.Symbol))
        
        # Convert the set of symbols to a list
        list_symbols = list(all_symbols)
        
        # If the result is zero, return an empty list
        if list_symbols == [0]:
            list_symbols = []
        
        params_in_mat = sorted(list_symbols,key=lambda x: str(x))
        return params_in_mat
        
        
    def get_rank_mass_matrix(self,mass_matrix, parameters):
        """
        Helper function to calculate the rank of mass matrices using random numbers between 0.1 and 1 for the parameters.
        
        Inputs:
             - mass_matrix: sympy matrix
                   The mass matrix to calculate its ranks.
             - parameters: list paramerters obtained from get_lagrangian()
             
        Outsputs:
             - mass_rank: int
                   Rank of mass_matrix
        """
        
        all_params = [p[0] for p in parameters]
        test_replacement_list = {a: random.uniform(0.1, 1) for a in all_params}
        mass_matrix_test = np.array(mass_matrix.xreplace(self.parametersSet).xreplace(test_replacement_list).evalf().tolist(), dtype=complex)
        mass_rank = np.linalg.matrix_rank(mass_matrix_test)
        
        return mass_rank
        
    def count_eff_flavons(self, lagrangian):
        """
        Function to count flavons present in the lagrangian.
            
        Inputs:
        ------
             -lagrangian: sympy expression
                   This is obtained from get_lagrangian()
                   
        Outputs:
        -------
             - eff_flavons: int
                   Number of effective flavons present in the lagrangian
             - flavons_not_present: list of strings
                   List of strings with the flavons not present in the lagrangian
        
        
        """
        
        
        eff_flavons = 0
        flavons_Not_present = []
        
        for i in range(1, self.NumberOfTotalFlavons  + 1):
            if f"phi{i}" in str(lagrangian):
                eff_flavons = eff_flavons + 1
            else:
                flavons_Not_present.append(f"phi{i}")

        return eff_flavons, flavons_Not_present
        
        
        
    def _extract_symbols(self, obj, symbolToFind):
        
        """
        Helper fuction to extract symbols from give a symbol to find and flatten them in a list.
        Inputs:
        -------
            - obj: list of symbols
                    List to look for symbols that contain symbolToFind
            - symbolToFind: str
                    this is the symbol that will be found in the obj.
                    
        Outputs:
        --------
        
            - founds_symbols: list of symbols
                    List of symbols that contain symbolToFind
        
        """
            
            
        founds_symbols = []
        if isinstance(obj, Symbol):
            if symbolToFind in obj.name:
                founds_symbols.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                founds_symbols.extend(self._extract_symbols(item, symbolToFind))
        return founds_symbols

        
    def get_mass_matrix(self, lagrangian ,parameters, left_fields, right_fields = None, higgs_fields = None,  factor_out = True):
    
        """
        This function obtains the sympy mass matrix for the lagrangian in terms of parameters. The mass matrix can be between a Dirac term in which case you need to specify the left_fields, the right_fields and the higgs_fields. You can also use it for a Majorana mass matrix, in which case you only specify left_fields.
        
        Inputs:
        -------
        
            - lagrangian: sympy symbol
                Lagrangian obtained from get_lagrangian.
            - parameters: list of sympy symbols
                Parameters obtained from get_lagrangian
            - left_fields: 2D list of strings
                Left fields that will be used for calculating the mass matrix
            - right_fields: 2D list of strings (optional, None by defaultu)
                Right fields that will be used for calculating the mass matrix
            - higgs_fields: 2D list of strings (optional, None by default)
                Higgs fields that will be used for calculating the mass matrix
            - factor_out: Boolean (True, by default)
                If true, it factors out one of the coefficients of the mass matrix. This can always be done for fitting observabbles. It makes the fit more efficiently
        
        
        Outputs:
        --------
        
            - mass_matrix: sympy matrix
                Dirac or Majorana mass matrix obtained from the lagrangian.
        
        """
        
        
        if left_fields not in self.particle_list_grouping:
            raise ValueError("left_fields must be in particle_list_grouping")

        if (right_fields is not None) and (right_fields not in self.particle_list_grouping):
            raise ValueError("right_fields must be in particle_list_grouping")

        if (higgs_fields is not None) and (higgs_fields not in self.particle_list_grouping):
            raise ValueError("higgs_fields must be in particle_list_grouping")
        
        
        minimum = {}
        left_fields_list = self.particle_dict[left_fields[0]]
        left_fields_flattened = self._extract_symbols(left_fields_list, left_fields[1])
        
        right_fields_list = self.particle_dict[right_fields[0]]
        right_fields_flattened = self._extract_symbols(right_fields_list, right_fields[1])
        
        if (higgs_fields is not None) and (right_fields is not None):
            higgs_fields_list = self.particle_dict[higgs_fields[0]]
            higgs_fields_flattened = self._extract_symbols(higgs_fields_list, higgs_fields[1])
            minimum.update({symbol: 1 for symbol in higgs_fields_flattened})
            self.parametersSet[higgs_fields_flattened[0]] = 1

        
        
        
        minimum.update({symbol: 0 for symbol in left_fields_flattened})
        minimum.update({symbol: 0 for symbol in right_fields_flattened})
        



        if lagrangian == 0:
            mass_matrix = sp.zeros(3, 3)
        else:
            mass_matrix = sp.Matrix(3, 3, lambda i, j: lagrangian.diff(left_fields_flattened[i], right_fields_flattened[j])).xreplace(minimum).xreplace(self.parametersSet)
            mass_matrix  = self.group._eliminate_small_real_imag(sp.simplify(mass_matrix))

        if factor_out:
            if len(self.ParametersInAMatrix(mass_matrix)) > 0:
                self.parametersSet[self.ParametersInAMatrix(mass_matrix)[0]] =  1


        mass_matrix = self.group._eliminate_small_real_imag(mass_matrix.xreplace(self.parametersSet))

        return mass_matrix


def make_latex_tableA4(file_path = None, model = None):


    """
    Reads a A4 model file and produce a LaTeX model table.

    Parameters
    ----------
    model_file : str
        Path to the file.
    model: dictionary
        Dictionary containing the file

    Returns
    -------
    str
        A LaTeX table environment as a string.
    """

    if file_path is not None:
        with open(file_path, "r") as file:
            data = json.load(file)

    if model is not None:
        data = model
    
    if file_path is None and model is None:
        raise ValueError("You need to give a model file or model")
        
    
    # Extract necessary components
    representations = data["representations"]
    particles = data["particles"]
    ZN_symmetry = data["ZN_symmetries"][0]
    fixed_flavon_vevs = data.get("fixed_flavon_vevs", [])
    
    # Format particle names with LaTeX subscripts
    def format_particle(name):
        if name == "Hu":
            return "H_u"
        elif name == "Hd":
            return "H_d"
        elif "phi" in name:
            name = name.replace("phi", "\\phi")
        for i, c in enumerate(name):
            if c.isdigit():
                return f"{name[:i]}_{name[i:]}"
        return f"{name}"

    # VEV mapping dictionary
    vev_map = {
        2: r"\left( \begin{array}{c} 1 \\ 0 \\ 0 \end{array} \right)",
        3: r"\left( \begin{array}{c} 1 \\ 1 \\ 1 \end{array} \right)",
        4: r"\left( \begin{array}{c} 0 \\ 1 \\ 0 \end{array} \right)",
        5: r"\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right)",
        6: r"\left(\begin{array}{c} 0 \\ 1 \\ -1 \end{array}\right)",
        7: r"\left(\begin{array}{c} 1 \\ \omega^2 \\ \omega \end{array}\right)",
        8: r"\left(\begin{array}{c} 1 \\ \omega \\ \omega^2 \end{array}\right)",
     }

    # Build lists only for non-zero representation entries
    filtered_particles = []
    filtered_representations = []
    filtered_indices = []

    for idx, (p, r) in enumerate(zip(particles[:16], representations[:16])):
        if any(r):  # skip if all entries are zero
            filtered_particles.append(format_particle(p))
            filtered_representations.append(r)
            filtered_indices.append(idx)

    latex_lines = []

    # Header row (particle names)
    header_row = " & " + " & ".join(filtered_particles) + " \\\\"
    latex_lines.append("\\hline")
    latex_lines.append(header_row)
    latex_lines.append("\\hline")

    # Second row (A4 representations)
    a4_row = ["A_4"]
    for rep in filtered_representations:
        if rep[0] == 1:
            a4_row.append("\\mathbf{1}")
        elif rep[1] == 1:
            a4_row.append("\\mathbf{1'}")
        elif rep[2] == 1:
            a4_row.append("\\mathbf{1''}")
        elif rep[3] >= 1:
            a4_row.append("\\mathbf{3}")
        else:
            a4_row.append("")
    latex_lines.append(" & ".join(a4_row) + " \\\\")
    latex_lines.append("\\hline")

    # Third row (\mathbb{Z}_N charges)
    zn_row = [f"\\mathbb{{Z}}_{ZN_symmetry}"]
    for rep in filtered_representations:
        zn_row.append(str(rep[4] - 1))
    latex_lines.append(" & ".join(zn_row) + " \\\\")
    latex_lines.append("\\hline")


    # Fourth row (Vevs)
    vev_row = [r"\text{Vevs}"]
    for idx, name in zip(filtered_indices, particles[:16]):
        if "phi" in name and idx >= 10:
            vev_row.append(vev_map.get(fixed_flavon_vevs[idx-11], ""))
        else:
            vev_row.append("")
    latex_lines.append(" & ".join(vev_row) + " \\\\")
    latex_lines.append("\\hline")

    n_columns = len(filtered_particles) + 1
    latex_code = "\\begin{array}{|" + "|".join(["c"] * n_columns) + "|}\n" + "\n".join(latex_lines) + "\n\\end{array}"
    
    return latex_code
        
        
def make_latex_tableT19(file_path = None, model = None):
    """
    Reads a T19 model file and produce a LaTeX model table.

    Parameters
    ----------
    model_file : str
        Path to the file.
    model: dictionary
        Dictionary containing the file
        
    
    Returns
    -------
    str
        A LaTeX array environment as a string.
    """

    if file_path is not None:
        with open(file_path, "r") as file:
            data = json.load(file)

    if model is not None:
        data = model
    
    if file_path is None and model is None:
        raise ValueError("You need to give a model file or model")
        

    # Extract necessary components
    representations = data["representations"]
    particles = data["particles"]
    ZN_symmetry = data["ZN_symmetries"][0]
    fixed_flavon_vevs = data.get("fixed_flavon_vevs", [])

    # Format particle names with LaTeX subscripts
    def format_particle(name: str) -> str:
        if name == "Hu":
            return "H_u"
        elif name == "Hd":
            return "H_d"
        elif "phi" in name:
            name = name.replace("phi", "\\phi")
        for i, c in enumerate(name):
            if c.isdigit():
                return f"{name[:i]}_{name[i:]}"
        return f"{name}"

    # VEV mapping dictionary
    vev_map = {
        2: r"\left( \begin{array}{c} 1 \\ 0 \\ 0 \end{array} \right)",
        3: r"\left( \begin{array}{c} 1 \\ 1 \\ 1 \end{array} \right)",
        4: r"\left( \begin{array}{c} 0 \\ 1 \\ 0 \end{array} \right)",
        5: r"\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right)",
        6: r"\left(\begin{array}{c} 0 \\ 1 \\ -1 \end{array}\right)",
        7: r"\left(\begin{array}{c} 1 \\ \omega^2 \\ \omega \end{array}\right)",
        8: r"\left(\begin{array}{c} 1 \\ \omega \\ \omega^2 \end{array}\right)",
    }

    # Build lists only for non-zero representation entries
    filtered_particles = []          # formatted names
    filtered_representations = []    # representation vectors
    filtered_indices = []            # original indices in 'particles'

    for idx, (p, r) in enumerate(zip(particles[:17], representations[:17])):
        if any(r):  # keep if any non-zero entry
            filtered_particles.append(format_particle(p))
            filtered_representations.append(r)
            filtered_indices.append(idx)

    latex_lines = []

    # Header row (particle names)
    header_row = " & " + " & ".join(filtered_particles) + " \\\\"
    latex_lines.append("\\hline")
    latex_lines.append(header_row)
    latex_lines.append("\\hline")

    # Second row (A4 representations)
    a4_row = ["T_{19}"]
    for rep in filtered_representations:
        if rep[0] == 1:
            a4_row.append("\\mathbf{1}")
        elif rep[1] == 1:
            a4_row.append("\\mathbf{1'}")
        elif rep[2] == 1:
            a4_row.append("\\mathbf{1''}")
        elif rep[3] >= 1:
            a4_row.append("\\mathbf{3_1}")
        elif rep[4] >= 1:
            a4_row.append("\\mathbf{\\overline{3}}_1")
        elif rep[5] >= 1:
            a4_row.append("\\mathbf{3_2}")
        elif rep[6] >= 1:
            a4_row.append("\\mathbf{3_3}")
        elif rep[7] >= 1:
            a4_row.append("\\mathbf{\\overline{3}}_2")
        elif rep[8] >= 1:
            a4_row.append("\\mathbf{\\overline{3}}_3")
        else:
            a4_row.append("")
    latex_lines.append(" & ".join(a4_row) + " \\\\")
    latex_lines.append("\\hline")

    # Third row (\mathbb{Z}_N charges)
    zn_row = [f"\\mathbb{{Z}}_{ZN_symmetry}"]
    for rep in filtered_representations:
        zn_row.append(str(rep[9] - 1))
    latex_lines.append(" & ".join(zn_row) + " \\\\")
    latex_lines.append("\\hline")

    # Fourth row (VEVs)
    vev_row = [r"\text{Vevs}"]
    # Maintain the same column order as filtered_particles by iterating filtered_indices
    for idx in filtered_indices:
        name_original = particles[idx]
        if "phi" in name_original and idx >= 10:
            j = idx - 11  # guard against OOB
            code = fixed_flavon_vevs[j] if 0 <= j < len(fixed_flavon_vevs) else None
            vev_row.append(vev_map.get(code, ""))
        else:
            vev_row.append("")
    latex_lines.append(" & ".join(vev_row) + " \\\\")
    latex_lines.append("\\hline")

    n_columns = len(filtered_particles) + 1
    latex_code = (
        "\\begin{array}{|"
        + "|".join(["c"] * n_columns)
        + "|}\n"
        + "\n".join(latex_lines)
        + "\n\\end{array}"
    )

    return latex_code


