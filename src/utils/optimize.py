import numpy as np
from scipy.optimize import bisect

# Definição de constantes globais
LCH3 = 0.2765  # Comprimento de um grupo metila (em nm)
LCH2 = 0.1265  # Comprimento de um grupo metileno (em nm)
NAV = 6.0221413e23  # Número de Avogadro
BOLTZMANN_CONST = 1.3806503e-23  # Constante de Boltzmann (em J/K)
PI = np.pi
ELEMENTARY_CHARGE = 1.6021766e-19  # Carga elementar (em C)
EPSILON_0 = 8.8541878e-12  # Permissividade elétrica do vácuo (em F/m)

class CMC_Calculator:
    """
    Classe para calcular a concentração micelar crítica (CMC) usando métodos numéricos.
    """

    def __init__(self, f_data):
        self.temperature = f_data['real_data'][0]  # Temperatura em Kelvin
        self.carbon_chain_length = f_data['real_data'][1]  # Comprimento da cadeia de carbono
        self.ethylene_oxide_units = f_data['real_data'][2]  # Unidades de óxido de etileno
        self.num_water_molecules = f_data['real_data'][3]  # Número de moléculas de água
        self.volume_core = f_data['real_data'][4]  # Volume do núcleo da micela
        self.num_surfactant_aggregates = f_data['real_data'][-1]  # Número de agregados de surfactantes

    def residue_function(self, surfactant_fraction, volume_fraction, segment_length, hydrophilic_volume,
                         surface_tension_water_surfactant, surface_tension_oil_surfactant, temperature,
                         core_volume, weight_fraction_water):
        """
        Função de resíduo usada para o método de bisseção.
        """
        log_term = -np.log(((surfactant_fraction / volume_fraction) ** (segment_length**3 / hydrophilic_volume)) /
                           ((1.0 - surfactant_fraction) / (1.0 - volume_fraction)))
        surface_energy_term = ((surface_tension_water_surfactant - surface_tension_oil_surfactant) /
                               (BOLTZMANN_CONST * temperature)) * (core_volume ** (2.0 / 3.0))
        interaction_term1 = (3.0 / 4.0) * weight_fraction_water * (1.0 - 2.0 * volume_fraction)
        interaction_term2 = -(1.0 / 2.0) * weight_fraction_water * (1.0 - 2.0 * surfactant_fraction)

        return log_term + surface_energy_term + interaction_term1 + interaction_term2

    def calculate_cmc(self, parameters):
        """
        Calcula a concentração micelar crítica (CMC) para um conjunto de parâmetros.
        """
        # Inicializações dos parâmetros de entrada
        aggregation_number = parameters[0]  # Número de agregação
        num_micelles = parameters[1]  # Número de micelas
        effective_diameter = parameters[2]  # Diâmetro efetivo
        free_energy = 1e5  # Energia livre inicial alta (para otimização)

        if aggregation_number * num_micelles < self.num_surfactant_aggregates:
            segment_length = 0.46e-9  # Comprimento do segmento (em m)
            ethylene_volume = 0.0631e-27  # Volume de óxido de etileno (em m³)
            hydrophilic_volume = self.ethylene_oxide_units * ethylene_volume
            initial_area = segment_length * segment_length
            area = initial_area
            weight_fraction_water = 1.2056 - 260.69 / self.temperature

            # Comprimento estendido da cauda surfactante (em metros)
            tail_length = ((self.carbon_chain_length - 1) * LCH2 + LCH3) * 1e-9
            num_segments = tail_length / segment_length

            # Volume molar dos grupos metileno e metila
            vch2 = 0.0269 + 1.46e-5 * (self.temperature - 298)
            vch3 = 0.0546 + 1.24e-4 * (self.temperature - 298)

            # Densidade da água e volume molar da água
            water_density = 999.65 + 2.0438e-1 * (self.temperature - 273.15) - 6.174e-2 * (self.temperature - 273.15)**1.5
            water_volume = (18.0 / (water_density * NAV)) * 1e-3

            # Raio da micela
            micelle_radius = (3 * self.volume_core * aggregation_number / (4 * PI))**(1.0 / 3.0)
            area_per_molecule = 4.0 * PI * micelle_radius**2 / aggregation_number

            packing_fraction = 1.0 / 3.0
            volume_fraction = hydrophilic_volume / (effective_diameter * area_per_molecule)

            if volume_fraction < 1.0 and area_per_molecule > initial_area:
                molecular_weight = self.carbon_chain_length * 12.0 + self.carbon_chain_length * 2.0 + 1.0
                surface_tension_water = 72.0 - 0.16 * (self.temperature - 298)
                surface_tension_surfactant = 35.0 - 0.098 * (self.temperature - 298) - 325 * molecular_weight**(-2.0 / 3.0)
                surface_tension_ethylene = 42.5 - 19.0 * (self.ethylene_oxide_units**(-2.0 / 3.0)) - 0.098 * (self.temperature - 293)

                surface_tension_water_surfactant = (surface_tension_surfactant + surface_tension_water -
                                                    2.0 * 0.55 * (surface_tension_surfactant * surface_tension_water)**0.5) * 1e-3
                surface_tension_oil_surfactant = (surface_tension_surfactant + surface_tension_ethylene -
                                                  2.0 * 0.55 * (surface_tension_surfactant * surface_tension_ethylene)**0.5) * 1e-3

                # Usando o método da bisseção para encontrar o volume fracionário da micela (fi_s)
                micelle_volume_fraction = bisect(self.residue_function, 1e-5, 0.9999,
                                                 args=(volume_fraction, segment_length, hydrophilic_volume,
                                                       surface_tension_water_surfactant, surface_tension_oil_surfactant,
                                                       self.temperature, self.volume_core, weight_fraction_water))

                log_ratio_term = np.log((1.0 - micelle_volume_fraction) / (1.0 - volume_fraction))
                volume_difference_term = (1.0 - (segment_length**3) / hydrophilic_volume) * (micelle_volume_fraction - volume_fraction)
                water_interaction_term = weight_fraction_water * ((1.0 / 2.0) * (micelle_volume_fraction**2) - (3.0 / 4.0) * (volume_fraction**2))

                aggregated_surface_tension = (surface_tension_water_surfactant + BOLTZMANN_CONST * self.temperature *
                                              self.volume_core**(-2.0 / 3.0) * (log_ratio_term + volume_difference_term + water_interaction_term))

                transfer_energy_ch2 = 5.85 * np.log(self.temperature) + (896.0 / self.temperature) - 36.15 - 0.0056 * self.temperature
                transfer_energy_ch3 = 3.38 * np.log(self.temperature) + (4064.0 / self.temperature) - 44.13 + 0.02595 * self.temperature

                free_energy_segments = np.zeros(7)
                free_energy_segments[0] = -1.4066076342 * self.carbon_chain_length - 7.579768587
                free_energy_segments[1] = (9.0 * packing_fraction * PI**2 / 80.0) * (micelle_radius**2 / (num_segments * segment_length**2))
                free_energy_segments[2] = -np.log(1 - (initial_area / area_per_molecule))
                free_energy_segments[3] = (aggregated_surface_tension / (BOLTZMANN_CONST * self.temperature)) * (area_per_molecule - initial_area)
                free_energy_segments[4] = (volume_fraction * hydrophilic_volume / (segment_length**3)) * ((0.5 - weight_fraction_water) /
                                                                                                          (1.0 + effective_diameter / micelle_radius))
                free_energy_segments[5] = 0.5 * ((effective_diameter**2 *
