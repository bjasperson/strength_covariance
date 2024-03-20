data import:
Initial shape: (163, 82)
initial factor list = 66
removed 4 very sparse columns: ['bulk_modulus_diamond', 'c44_diamond', 'c12_diamond', 'c11_diamond']
removed 23 empty columns: ['thermal_expansion_coeff_bcc', 'thermal_expansion_coeff_diamond', 'surface_energy_100_bcc', 'surface_energy_110_bcc', 'surface_energy_111_bcc', 'surface_energy_121_bcc', 'ideal_surface_energy_100_bcc', 'ideal_surface_energy_110_bcc', 'ideal_surface_energy_111_bcc', 'ideal_surface_energy_121_bcc', 'relaxed_formation_potential_energy_bcc', 'relaxed_formation_potential_energy_diamond', 'relaxed_formation_potential_energy_hcp', 'unrelaxed_formation_potential_energy_bcc', 'unrelaxed_formation_potential_energy_diamond', 'unrelaxed_formation_potential_energy_hcp', 'vacancy_migration_energy_bcc', 'vacancy_migration_energy_diamond', 'vacancy_migration_energy_hcp', 'vacancy_migration_energy_sc', 'relaxation_volume_bcc', 'relaxation_volume_diamond', 'relaxation_volume_hcp']
cleaned data
final factor count is 39: ['lattice_constant_bcc', 'lattice_constant_diamond', 'lattice_constant_fcc', 'lattice_constant_sc', 'bulk_modulus_bcc', 'bulk_modulus_fcc', 'bulk_modulus_sc', 'c44_bcc', 'c44_fcc', 'c44_sc', 'c12_bcc', 'c12_fcc', 'c12_sc', 'c11_bcc', 'c11_fcc', 'c11_sc', 'cohesive_energy_bcc', 'cohesive_energy_diamond', 'cohesive_energy_fcc', 'cohesive_energy_sc', 'thermal_expansion_coeff_fcc', 'surface_energy_100_fcc', 'surface_energy_110_fcc', 'surface_energy_111_fcc', 'surface_energy_121_fcc', 'ideal_surface_energy_100_fcc', 'ideal_surface_energy_110_fcc', 'ideal_surface_energy_111_fcc', 'ideal_surface_energy_121_fcc', 'extr_stack_fault_energy_fcc', 'intr_stack_fault_energy_fcc', 'unstable_stack_energy_fcc', 'unstable_stack_energy_slip_fraction_fcc', 'unstable_twinning_energy_fcc', 'unstable_twinning_energy_slip_fraction_fcc', 'relaxed_formation_potential_energy_fcc', 'unrelaxed_formation_potential_energy_fcc', 'vacancy_migration_energy_fcc', 'relaxation_volume_fcc']
shuffled
29 factors: Index(['c12_sc', 'lattice_constant_bcc', 'unstable_stack_energy_fcc',
       'c12_fcc', 'extr_stack_fault_energy_fcc', 'bulk_modulus_bcc', 'c44_fcc',
       'surface_energy_100_fcc', 'c12_bcc', 'cohesive_energy_fcc', 'c44_sc',
       'c11_fcc', 'intr_stack_fault_energy_fcc', 'cohesive_energy_sc',
       'cohesive_energy_bcc', 'cohesive_energy_diamond',
       'unstable_stack_energy_slip_fraction_fcc', 'c44_bcc',
       'unstable_twinning_energy_fcc', 'lattice_constant_fcc',
       'lattice_constant_sc', 'unstable_twinning_energy_slip_fraction_fcc',
       'c11_sc', 'thermal_expansion_coeff_fcc', 'bulk_modulus_fcc',
       'lattice_constant_diamond', 'bulk_modulus_sc',
       'ideal_surface_energy_100_fcc', 'c11_bcc'],
      dtype='object')

corr_list = ['c44_fcc', 'lattice_constant_bcc', 'c11_fcc', 'lattice_constant_fcc', 'surface_energy_100_fcc', 'ideal_surface_energy_100_fcc', 'lattice_constant_sc', 'c11_sc', 'unstable_twinning_energy_fcc', 'cohesive_energy_fcc', 'cohesive_energy_bcc', 'unstable_stack_energy_fcc', 'bulk_modulus_fcc', 'lattice_constant_diamond', 'cohesive_energy_sc', 'c12_fcc', 'c44_bcc', 'bulk_modulus_sc', 'thermal_expansion_coeff_fcc', 'cohesive_energy_diamond', 'c44_sc', 'c12_sc', 'unstable_stack_energy_slip_fraction_fcc', 'intr_stack_fault_energy_fcc', 'unstable_twinning_energy_slip_fraction_fcc', 'extr_stack_fault_energy_fcc', 'c12_bcc', 'bulk_modulus_bcc', 'c11_bcc']

29 all factors model: Index(['c44_fcc', 'lattice_constant_bcc', 'c11_fcc', 'lattice_constant_fcc',
       'surface_energy_100_fcc', 'ideal_surface_energy_100_fcc',
       'lattice_constant_sc', 'c11_sc', 'unstable_twinning_energy_fcc',
       'cohesive_energy_fcc', 'cohesive_energy_bcc',
       'unstable_stack_energy_fcc', 'bulk_modulus_fcc',
       'lattice_constant_diamond', 'cohesive_energy_sc', 'c12_fcc', 'c44_bcc',
       'bulk_modulus_sc', 'thermal_expansion_coeff_fcc',
       'cohesive_energy_diamond', 'c44_sc', 'c12_sc',
       'unstable_stack_energy_slip_fraction_fcc',
       'intr_stack_fault_energy_fcc',
       'unstable_twinning_energy_slip_fraction_fcc',
       'extr_stack_fault_energy_fcc', 'c12_bcc', 'bulk_modulus_bcc',
       'c11_bcc'],
      dtype='object')

3 factor model: Index(['c44_fcc', 'extr_stack_fault_energy_fcc',
       'vacancy_migration_energy_fcc'],
      dtype='object')
