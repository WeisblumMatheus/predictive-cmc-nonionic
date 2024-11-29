import numpy as np
from scipy.optimize import bisect, minimize, root_scalar
from pyswarm import pso

from src.cmc_calculations.cmc_start import read_dat_file_values

# Definição de constantes globais
LCH3 = 0.2765  # Comprimento de um grupo metila (em nm)
LCH2 = 0.1265  # Comprimento de um grupo metileno (em nm)
NAV = 6.0221413e23  # Número de Avogadro
BOLTZMANN_CONST = 1.3806503e-23  # Constante de Boltzmann (em J/K)
PI = np.pi
ELEMENTARY_CHARGE = 1.6021766e-19  # Carga elementar (em C)
EPSILON_0 = 8.8541878e-12  # Permissividade elétrica do vácuo (em F/m)


class CmcCalculator:
    """
    Classe para calcular a concentração micelar crítica (CMC) usando métodos numéricos.
    """

    @staticmethod
    def residue_function(surfactant_fraction, volume_fraction, segment_length, hydrophilic_volume,
                         surface_tension_water_surfactant, surface_tension_oil_surfactant, temperature,
                         core_volume, flory_huggins_type_interaction):

        """
        Função de resíduo usada para o método de bisseção.
        """
        log_term = -np.log(((surfactant_fraction / volume_fraction) ** (segment_length ** 3 / hydrophilic_volume)) /
                           ((1.0 - surfactant_fraction) / (1.0 - volume_fraction)))
        surface_energy_term = ((surface_tension_water_surfactant - surface_tension_oil_surfactant) /
                               (BOLTZMANN_CONST * temperature)) * (core_volume ** (2.0 / 3.0))
        interaction_term1 = (3.0 / 4.0) * flory_huggins_type_interaction * (1.0 - 2.0 * volume_fraction)
        interaction_term2 = -(1.0 / 2.0) * flory_huggins_type_interaction * (1.0 - 2.0 * surfactant_fraction)

        return log_term + surface_energy_term + interaction_term1 + interaction_term2

    def calculate_cmc(self, parameters, f_data):
        """
        Calcula a concentração micelar crítica (CMC) para um conjunto de parâmetros.
        """
        temperature = f_data[0]  # Temperatura em Kelvin
        carbon_chain_length = f_data[1]  # Comprimento da cadeia de carbono
        ethylene_oxide_units = f_data[2]  # Unidades de óxido de etileno
        num_water_molecules = f_data[3]  # Número de moléculas de água
        volume_core = f_data[4]  # Volume do núcleo da micela
        num_surfactant_aggregates = f_data[-1]  # Número de agregados de surfactantes

        # Inicializações dos parâmetros de entrada
        free_gibbs_energy = 1e5
        aggregation_number = parameters[0]  # Número de agregação
        num_micelles = parameters[1]  # Número de micelas
        effective_diameter = parameters[2]  # Diâmetro efetivo

        if aggregation_number * num_micelles < num_surfactant_aggregates:
            segment_length = 0.46e-9  # Comprimento do segmento (em m)
            ethylene_volume = 0.0631e-27  # Volume de óxido de etileno (em m³)
            hydrophilic_volume = ethylene_oxide_units * ethylene_volume
            initial_area = segment_length * segment_length
            flory_huggins_type_interaction = 1.2056 - 260.69 / temperature

            # Comprimento estendido da cauda surfactante (em metros)
            tail_length = ((carbon_chain_length - 1) * LCH2 + LCH3) * 1e-9
            num_segments = tail_length / segment_length

            # Raio da micela
            micelle_radius = (3 * volume_core * aggregation_number / (4 * PI)) ** (1.0 / 3.0)
            area_per_molecule = 4.0 * PI * micelle_radius ** 2 / aggregation_number

            packing_fraction = 1.0 / 3.0
            volume_fraction = hydrophilic_volume / (effective_diameter * area_per_molecule)

            if volume_fraction < 1.0 and area_per_molecule > initial_area:
                molecular_weight = carbon_chain_length * 12.0 + carbon_chain_length * 2.0 + 1.0
                surface_tension_water = 72.0 - 0.16 * (temperature - 298)
                surface_tension_surfactant = 35.0 - 0.098 * (temperature - 298) - 325 * molecular_weight ** (
                        -2.0 / 3.0)
                surface_tension_ethylene = 42.5 - 19.0 * (ethylene_oxide_units ** (-2.0 / 3.0)) - 0.098 * (
                        temperature - 293)

                surface_tension_water_surfactant = (surface_tension_surfactant + surface_tension_water -
                                                    2.0 * 0.55 * (
                                                            surface_tension_surfactant * surface_tension_water) ** 0.5) * 1e-3
                surface_tension_oil_surfactant = (surface_tension_surfactant + surface_tension_ethylene -
                                                  2.0 * 0.55 * (
                                                          surface_tension_surfactant * surface_tension_ethylene) ** 0.5) * 1e-3

                # micelle_volume_fraction = bisect(self.residue_function, 1e-5, 0.9999,
                #                                  args=(volume_fraction, segment_length, hydrophilic_volume,
                #                                        surface_tension_water_surfactant, surface_tension_oil_surfactant,
                #                                        temperature, volume_core, flory_huggins_type_interaction),
                #                                  maxiter=100, xtol=1.e-9)
                result = root_scalar(
                    self.residue_function,
                    args=(volume_fraction, segment_length, hydrophilic_volume,
                          surface_tension_water_surfactant, surface_tension_oil_surfactant,
                          temperature, volume_core, flory_huggins_type_interaction),
                    method='newton',
                    x0=0.9,  # Chute inicial
                    xtol=1.e-9,
                    maxiter=100
                )
                micelle_volume_fraction = result.root

                log_ratio_term = np.log((1.0 - micelle_volume_fraction) / (1.0 - volume_fraction))
                volume_difference_term = (1.0 - (segment_length ** 3) / hydrophilic_volume) * (
                        micelle_volume_fraction - volume_fraction)
                water_interaction_term = flory_huggins_type_interaction * (
                        (1.0 / 2.0) * (micelle_volume_fraction ** 2) - (3.0 / 4.0) * (volume_fraction ** 2))

                aggregated_surface_tension = (surface_tension_water_surfactant + BOLTZMANN_CONST * temperature *
                                              volume_core ** (-2.0 / 3.0) * (
                                                      log_ratio_term + volume_difference_term + water_interaction_term))

                free_energy_segments = np.zeros(7)
                free_energy_segments[0] = -1.4066076342 * carbon_chain_length - 7.579768587
                free_energy_segments[1] = (9.0 * packing_fraction * PI ** 2 / 80.0) * (
                        micelle_radius ** 2 / (num_segments * segment_length ** 2))
                free_energy_segments[2] = -np.log(1 - (initial_area / area_per_molecule))
                free_energy_segments[3] = (aggregated_surface_tension / (BOLTZMANN_CONST * temperature)) * (
                        area_per_molecule - initial_area)
                free_energy_segments[4] = (volume_fraction * hydrophilic_volume / (segment_length ** 3)) * (
                        (0.5 - flory_huggins_type_interaction) /
                        (1.0 + effective_diameter / micelle_radius))
                free_energy_segments[5] = 0.5 * ((effective_diameter ** 2 * segment_length / hydrophilic_volume) +
                                                 (2.0 * (hydrophilic_volume ** 0.5) / (
                                                         effective_diameter * segment_length ** 0.5)) - 3.0)
                free_energy_segments[6] = np.sum(free_energy_segments[:6]) * aggregation_number * num_micelles

                free_gibbs_energy = self.calculate_gibbs_energy(num_micelles, aggregation_number, free_energy_segments,
                                                                num_water_molecules, num_surfactant_aggregates)

        return free_gibbs_energy

    def calculate_gibbs_energy(self, num_micelles, aggregation_number, free_energy_segments, num_water_molecules,
                               num_surfactant_aggregates):
        """
        Calcula a energia livre de Gibbs para um conjunto de parâmetros.
        """
        remaining_surfactants = num_surfactant_aggregates - num_micelles * aggregation_number
        total_molecules = (num_water_molecules + remaining_surfactants + num_micelles)
        water_fraction = num_water_molecules / total_molecules
        free_surfactant_fraction = remaining_surfactants / total_molecules
        micelle_fraction = num_micelles / total_molecules

        free_energy_mixture = np.zeros(4)
        free_energy_mixture[0] = num_water_molecules * np.log(water_fraction)
        free_energy_mixture[1] = remaining_surfactants * np.log(free_surfactant_fraction)
        free_energy_mixture[2] = num_micelles * np.log(micelle_fraction)
        free_energy_mixture[3] = free_energy_mixture[0] + free_energy_mixture[1] + free_energy_mixture[2]

        # Cálculo final da energia livre
        free_energy = free_energy_mixture[3] + free_energy_segments[6]

        return free_energy

    def cmc_fun(self, x, f_data_in_x, f_data_in_y):
        """
        Função objetivo para otimização da CMC.
        """
        small = 1e-6
        y = 0.0

        for i in range(len(f_data_in_x)):
            Nsatemp = f_data_in_x[i]
            N1aTemp = f_data_in_y[i]

            term1 = x[0] * Nsatemp + x[1]
            term2 = x[2] * (Nsatemp - x[3])
            term3 = (1 + (Nsatemp - x[3]) / (np.sqrt((Nsatemp - x[3]) ** 2 + (small ** 0.5) ** 2)))

            y += (term1 - term2 * term3 - N1aTemp) ** 2

        return y

    def optim_cmc(self):
        """
        Otimiza a concentração micelar crítica (CMC) usando PSO e Nelder-Mead.
        """

        f_data = read_dat_file_values("/Users/weisblum/Documents/TratumProject/cmc_non_ionic/input_non.dat")

        nsa_i = 10.0
        nsa_f = f_data[6]
        delta_n = f_data[5]
        n = 3

        plim = np.array([
            [10.0, 800.0],
            [0.000001, 800.0],
            [0.0001e-9, 5e-8]
        ])

        n_nsa = int((nsa_f - nsa_i) / delta_n) + 1
        nw = f_data[3]
        x = np.zeros(n_nsa)
        y = np.zeros(n_nsa)

        f_data_in_x = []
        f_data_in_y = []

        with open("cmc.dat", "w") as out_file:
            out_file.write(f"{int((nsa_f - nsa_i) / delta_n) + 2}\n")
            nsa = nsa_i
            f_data[4] *= 1e-27

            for i in range(n_nsa):
                f_data.append(nsa)

                # PSO
                lb = plim[:, 0]
                ub = plim[:, 1]

                point_optim, _ = pso(lambda x: self.calculate_cmc(x, f_data), lb, ub, swarmsize=200, maxiter=200,
                                     phig=1.5, omega=0.4, phip=1.5)
                print(f"pso: {point_optim}")

                # Criando pontos inciais para aplicar no Nelder-Mead (Simplex)
                p = np.zeros((n + 1, n))  # Inicialização do Simplex
                y_simplex = np.zeros(n + 1)
                p[0] = point_optim
                y_simplex[0] = self.calculate_cmc(point_optim, f_data)

                # Geração de pontos para o Simplex
                for j in range(1, n + 1):
                    p[j] = p[j - 1] * 1.25  # Escalonando os pontos
                    point_optim = p[j]
                    y_simplex[j] = self.calculate_cmc(point_optim, f_data)

                # Minimização com Nelder-Mead
                result = minimize(lambda x: self.calculate_cmc(x, f_data), p[np.argmin(y_simplex)],
                                  method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7, 'initial_simplex': p})
                print(f"Nelder-Mead result: {result.x}")

                point_optim = result.x

                x[i] = nsa
                y[i] = nsa - point_optim[0] * point_optim[1]

                f_data_in_x.append(x[i])
                f_data_in_y.append(y[i])
                print(f"{x[i]} {y[i]}")
                out_file.write(f"{x[i]} {y[i]}\n")

                nsa += delta_n

        # Segundo ciclo com diferentes limites
        n = 4
        plim = np.array([
            [0.0, 1.0],
            [0.0, 0.1],
            [0.0, 1.0],
            [0.0, nsa_f]
        ])

        # PSO
        lb = plim[:, 0]
        ub = plim[:, 1]
        point_optim, _ = pso(lambda x: self.cmc_fun(x, f_data_in_x, f_data_in_y), lb, ub, swarmsize=100, maxiter=50,
                             phig=1.5, omega=0.4, phip=1.4)

        # Criando pontos inciais para aplicar no Nelder-Mead
        p = np.zeros((n + 1, n))
        z = np.zeros(n + 1)
        p[0] = point_optim
        z[0] = self.cmc_fun(point_optim, f_data_in_x, f_data_in_y)

        # Geração de pontos para o Simplex
        for j in range(1, n + 1):
            p[j] = p[j - 1] * 1.05  # Escalonando os pontos
            point_optim = p[j]
            z[j] = self.cmc_fun(point_optim, f_data_in_x, f_data_in_y)

        # Minimização com Nelder-Mead
        result = minimize(lambda x: self.cmc_fun(x, f_data_in_x, f_data_in_y), p[np.argmin(z)], method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7, 'initial_simplex': p})
        point_optim = result.x

        cmc_out = (point_optim[3] / nw) * 1000.0 / 18.0
        print(f"CMC: {cmc_out}")

        with open("cmc.dat", "a") as out_file:
            out_file.write(f"{point_optim[3]} {cmc_out * 1000.0}\n")

        return cmc_out


optimizer = CmcCalculator()
results = optimizer.optim_cmc()
print(f"CMC Calculada: {results}")
