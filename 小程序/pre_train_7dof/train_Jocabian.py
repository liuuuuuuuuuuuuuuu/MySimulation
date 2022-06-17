import numpy as np
import random

import torch

input_data =[]
output_data = []
total_data = []
for i in range(100000):
    theta1 = random.uniform(-np.pi / 2, np.pi / 2)
    theta2 = random.uniform(-np.pi / 2, np.pi / 2)
    theta3 = random.uniform(-np.pi / 2, np.pi / 2)
    theta4 = random.uniform(-np.pi / 2, np.pi / 2)
    theta5 = random.uniform(-np.pi / 2, np.pi / 2)
    theta6 = random.uniform(-np.pi / 2, np.pi / 2)
    theta7 = random.uniform(-np.pi / 2, np.pi / 2)

    J11 = (np.cos(theta7) * (np.cos(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(theta4) * (
                    np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))) + np.sin(theta5) * (
                                                     np.cos(theta4) * (
                                                         np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(
                                                     theta3)) - np.sin(theta4) * (
                                                                 np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(
                                                             theta3) * np.sin(theta1)))) - np.sin(theta1) * np.sin(theta2) * np.sin(
        theta6)) + np.sin(theta7) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3)) - np.sin(theta4) * (
                    np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1))) - np.sin(theta5) * (
                                              np.cos(theta4) * (
                                                  np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(
                                              theta1)) + np.sin(theta4) * (
                                                          np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(
                                                      theta3))))) * (
                      (27 * np.cos(theta2) * np.sin(theta5) * np.sin(theta7)) / 20 + (
                          47 * np.cos(theta2) * np.sin(theta6) * np.sin(theta7)) / 100 + (
                                  37 * np.cos(theta7) * np.sin(theta2) * np.sin(theta6)) / 100 - (
                                  37 * np.cos(theta2) * np.cos(theta6) * np.cos(theta7)) / 100 - (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.sin(theta5) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.sin(theta4) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta3) * np.cos(theta7) * np.sin(theta2) * np.sin(theta6)) / 20 - (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7)) / 20 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.sin(theta4) * np.sin(theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3) * np.sin(theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta7)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta5)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4)) / 100 + (
                                  47 * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3)) / 100 + (
                                  27 * np.cos(theta3) * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta6)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta6) * np.cos(theta7) * np.sin(theta4) * np.sin(theta5)) / 20 - (
                                  37 * np.cos(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  111 * np.cos(theta3) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  111 * np.cos(theta4) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  111 * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta7)) / 100 - (
                                  47 * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                                  27 * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(
                              theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta6) * np.cos(theta7) * np.sin(theta4) * np.sin(
                              theta5)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta6) * np.cos(theta7) * np.sin(theta3) * np.sin(
                              theta5)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta3) * np.sin(
                              theta4)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta6)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta5)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta4)) / 100 + (
                                  111 * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta3)) / 100 - (
                                  47 * np.cos(theta3) * np.cos(theta6) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta3) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta6)) / 100 - (
                                  47 * np.cos(theta4) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(
                              theta6)) / 100 - (
                                  47 * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta6)) / 100 - (
                                  111 * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta5)) / 100) + (np.sin(theta7) * (np.cos(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(theta4) * (
                    np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))) + np.sin(theta5) * (
                                                                                           np.cos(theta4) * (
                                                                                               np.cos(theta1) * np.cos(
                                                                                           theta3) - np.cos(theta2) * np.sin(
                                                                                           theta1) * np.sin(theta3)) - np.sin(
                                                                                       theta4) * (np.cos(theta1) * np.sin(
                                                                                       theta3) + np.cos(theta2) * np.cos(
                                                                                       theta3) * np.sin(theta1)))) - np.sin(
        theta1) * np.sin(theta2) * np.sin(theta6)) - np.cos(theta7) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3)) - np.sin(theta4) * (
                    np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1))) - np.sin(theta5) * (
                                                                          np.cos(theta4) * (
                                                                              np.cos(theta1) * np.sin(theta3) + np.cos(
                                                                          theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(
                                                                      theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(
                                                                      theta2) * np.sin(theta1) * np.sin(theta3))))) * (
                      (37 * np.sin(theta2) * np.sin(theta6) * np.sin(theta7)) / 100 - (
                          27 * np.cos(theta2) * np.cos(theta7) * np.sin(theta5)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta6) * np.sin(theta7)) / 100 - (
                                  47 * np.cos(theta2) * np.cos(theta7) * np.sin(theta6)) / 100 - (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta7) * np.sin(theta5)) / 20 - (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.cos(theta7) * np.sin(theta4)) / 20 - (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta3) * np.sin(theta2) * np.sin(theta6) * np.sin(theta7)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta7) * np.sin(theta5)) / 100 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.cos(theta7) * np.sin(theta4)) / 100 - (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta3)) / 100 - (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2)) / 100 - (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta7)) / 20 + (
                                  37 * np.cos(theta2) * np.cos(theta7) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(theta5) * np.sin(theta7)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta5) * np.sin(theta2) * np.sin(theta4) * np.sin(theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                                  47 * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta7)) / 100 + (
                                  111 * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5)) / 100 + (
                                  111 * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4)) / 100 + (
                                  27 * np.cos(theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(theta6) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta6) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 20 - (
                                  47 * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  27 * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6) * np.sin(theta7)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(
                              theta7)) / 100 - (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(
                              theta2)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta6) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta6) * np.sin(theta3) * np.sin(theta5) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta6) * np.sin(
                              theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.sin(theta2) * np.sin(theta5) * np.sin(
                              theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta4) * np.sin(
                              theta7)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4) * np.sin(
                              theta5)) / 100 + (
                                  111 * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(
                              theta7)) / 100 + (
                                  47 * np.cos(theta4) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(
                              theta5)) / 100 + (
                                  47 * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(
                              theta4)) / 100 - (
                                  37 * np.cos(theta3) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta4) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(theta6) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6) * np.sin(
                              theta7)) / 100 - (
                                  111 * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta7)) / 100) + (np.sin(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(theta4) * (
                    np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))) + np.sin(theta5) * (
                                                                            np.cos(theta4) * (
                                                                                np.cos(theta1) * np.cos(theta3) - np.cos(
                                                                            theta2) * np.sin(theta1) * np.sin(theta3)) - np.sin(
                                                                        theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(
                                                                        theta2) * np.cos(theta3) * np.sin(theta1)))) + np.cos(
        theta6) * np.sin(theta1) * np.sin(theta2)) * ((27 * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4)) / 20 - (
                37 * np.cos(theta6) * np.sin(theta2)) / 100 - (27 * np.cos(theta3) * np.cos(theta6) * np.sin(theta2)) / 20 - (
                                                            27 * np.cos(theta2) * np.cos(theta5) * np.sin(theta6)) / 20 - (
                                                            27 * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.sin(
                                                        theta2)) / 20 - (
                                                            27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(
                                                        theta6)) / 20 - (37 * np.cos(theta2) * np.sin(theta6)) / 100 + (
                                                            27 * np.cos(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(
                                                        theta6)) / 20 - (
                                                            37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(
                                                        theta5) * np.sin(theta6)) / 100 - (
                                                            37 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(
                                                        theta6) * np.sin(theta2)) / 100 + (
                                                            37 * np.cos(theta2) * np.cos(theta3) * np.sin(theta4) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta2) * np.cos(theta4) * np.sin(theta3) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta2) * np.cos(theta5) * np.sin(theta3) * np.sin(
                                                        theta4) * np.sin(theta6)) / 100 + (
                                                            111 * np.cos(theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100 + (
                                                            111 * np.cos(theta3) * np.cos(theta5) * np.sin(theta2) * np.sin(
                                                        theta4) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta3) * np.cos(theta6) * np.sin(theta2) * np.sin(
                                                        theta4) * np.sin(theta5)) / 100 + (
                                                            111 * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(
                                                        theta3) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta4) * np.cos(theta6) * np.sin(theta2) * np.sin(
                                                        theta3) * np.sin(theta5)) / 100 + (
                                                            37 * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(
                                                        theta3) * np.sin(theta4)) / 100 - (
                                                            111 * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100)
    J12 = (np.sin(theta1) * (
                111 * np.cos(theta2) + 47 * np.cos(theta2) * np.cos(theta6) + 135 * np.sin(theta2) * np.sin(theta3) + 135 * np.cos(
            theta3) * np.sin(theta2) * np.sin(theta4) + 135 * np.cos(theta4) * np.sin(theta2) * np.sin(theta3) + 37 * np.cos(
            theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(theta5) + 37 * np.cos(theta3) * np.cos(theta5) * np.sin(theta2) * np.sin(
            theta4) + 37 * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) - 37 * np.sin(theta2) * np.sin(theta3) * np.sin(
            theta4) * np.sin(theta5) - 47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta6) + 47 * np.cos(
            theta3) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6) + 47 * np.cos(theta4) * np.sin(theta2) * np.sin(
            theta3) * np.sin(theta5) * np.sin(theta6) + 47 * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
            theta6))) / 100
    J13 = (27 * np.cos(theta2) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4)) / 20 - (
                27 * np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) / 20 - (
                      27 * np.cos(theta1) * np.cos(theta3) * np.sin(theta4)) / 20 - (
                      27 * np.cos(theta1) * np.cos(theta4) * np.sin(theta3)) / 20 - (
                      27 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta1)) / 20 - (
                      37 * np.cos(theta1) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5)) / 100 - (
                      37 * np.cos(theta1) * np.cos(theta3) * np.cos(theta5) * np.sin(theta4)) / 100 - (
                      37 * np.cos(theta1) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3)) / 100 - (
                      27 * np.cos(theta1) * np.sin(theta3)) / 20 + (
                      37 * np.cos(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                      37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta1)) / 100 + (
                      47 * np.cos(theta1) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta6)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta3) * np.sin(theta1) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta4) * np.sin(theta1) * np.sin(theta3) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta5) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta4) * np.sin(theta3) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta5) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta1) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.sin(theta1) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta1) * np.sin(theta3) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta2) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100
    J14 = (np.cos(theta7) * (np.cos(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(theta4) * (
                    np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))) + np.sin(theta5) * (
                                                     np.cos(theta4) * (
                                                         np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(
                                                     theta3)) - np.sin(theta4) * (
                                                                 np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(
                                                             theta3) * np.sin(theta1)))) - np.sin(theta1) * np.sin(theta2) * np.sin(
        theta6)) + np.sin(theta7) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3)) - np.sin(theta4) * (
                    np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1))) - np.sin(theta5) * (
                                              np.cos(theta4) * (
                                                  np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(
                                              theta1)) + np.sin(theta4) * (
                                                          np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(
                                                      theta3))))) * (
                      (27 * np.sin(theta5) * np.sin(theta7)) / 20 + (47 * np.sin(theta6) * np.sin(theta7)) / 100 - np.cos(theta7) * (
                          (37 * np.cos(theta6)) / 100 + (27 * np.cos(theta5) * np.cos(theta6)) / 20)) - (np.sin(theta7) * (
                np.cos(theta6) * (np.cos(theta5) * (
                    np.cos(theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(
                theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))) + np.sin(theta5) * (
                                           np.cos(theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(
                                       theta3)) - np.sin(theta4) * (
                                                       np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(
                                                   theta1)))) - np.sin(theta1) * np.sin(theta2) * np.sin(theta6)) - np.cos(
        theta7) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3)) - np.sin(theta4) * (
                    np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1))) - np.sin(theta5) * (
                               np.cos(theta4) * (
                                   np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(
                           theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))))) * (
                      (27 * np.cos(theta7) * np.sin(theta5)) / 20 + (47 * np.cos(theta7) * np.sin(theta6)) / 100 + np.sin(theta7) * (
                          (37 * np.cos(theta6)) / 100 + (27 * np.cos(theta5) * np.cos(theta6)) / 20)) - (np.sin(theta6) * (
                np.cos(theta5) * (
                    np.cos(theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(
                theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))) + np.sin(theta5) * (
                            np.cos(theta4) * (np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3)) - np.sin(
                        theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)))) + np.cos(
        theta6) * np.sin(theta1) * np.sin(theta2)) * ((37 * np.sin(theta6)) / 100 + (27 * np.cos(theta5) * np.sin(theta6)) / 20)
    J15 = (37 * np.cos(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                37 * np.cos(theta1) * np.cos(theta3) * np.cos(theta5) * np.sin(theta4)) / 100 - (
                      37 * np.cos(theta1) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3)) / 100 - (
                      37 * np.cos(theta1) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5)) / 100 - (
                      37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta1)) / 100 + (
                      47 * np.cos(theta1) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta6)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta3) * np.sin(theta1) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta4) * np.sin(theta1) * np.sin(theta3) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta5) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta4) * np.sin(theta3) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta5) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta1) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.sin(theta1) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta1) * np.sin(theta3) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta2) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100
    J16 = (47 * np.cos(theta1) * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.sin(theta5)) / 100 - (
                47 * np.sin(theta1) * np.sin(theta2) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta1) * np.cos(theta3) * np.cos(theta5) * np.cos(theta6) * np.sin(theta4)) / 100 + (
                      47 * np.cos(theta1) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta3)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta6) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                      47 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta1)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta3) * np.cos(theta6) * np.sin(theta1) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta4) * np.cos(theta6) * np.sin(theta1) * np.sin(theta3) * np.sin(theta5)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4)) / 100
    J17 = 0
    J21 = 0
    J22 = (27 * np.cos(theta2) * np.sin(theta3)) / 20 - (111 * np.sin(theta2)) / 100 - (47 * np.cos(theta6) * np.sin(theta2)) / 100 + (
                27 * np.cos(theta2) * np.cos(theta3) * np.sin(theta4)) / 20 + (
                      27 * np.cos(theta2) * np.cos(theta4) * np.sin(theta3)) / 20 + (
                      37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.sin(theta4)) / 100 + (
                      37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3)) / 100 - (
                      37 * np.cos(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                      47 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta2) * np.cos(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta2) * np.cos(theta4) * np.sin(theta3) * np.sin(theta5) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta2) * np.cos(theta5) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6)) / 100
    J23 = (np.sin(theta2) * (
                37 * np.cos(theta3 + theta4 + theta5) - (47 * np.cos(theta3 + theta4 + theta5 + theta6)) / 2 + 135 * np.cos(
            theta3 + theta4) + 135 * np.cos(theta3) + (47 * np.cos(theta3 + theta4 + theta5 - theta6)) / 2)) / 100;
    J24 = (np.sin(theta2) * (
                135 * np.cos(theta3 + theta4) - 37 * np.sin(theta3 + theta4) * np.sin(theta5) + 37 * np.cos(theta3 + theta4) * np.cos(
            theta5) + 47 * np.cos(theta3 + theta4) * np.sin(theta5) * np.sin(theta6) + 47 * np.sin(theta3 + theta4) * np.cos(
            theta5) * np.sin(theta6))) / 100;
    J25 = (np.sin(theta2) * (37 * np.cos(theta3 + theta4 + theta5) + 47 * np.sin(theta3 + theta4 + theta5) * np.sin(theta6))) / 100;
    J26 = (47 * np.cos(theta3) * np.cos(theta6) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta2)) / 100 - (
                      47 * np.cos(theta2) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta4) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5)) / 100 + (
                      47 * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4)) / 100;
    J27 = 0;
    J31 = - (np.sin(theta7) * (np.cos(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(theta4) * (
                    np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) + np.sin(theta5) * (
                                                       np.cos(theta3) * np.cos(theta4) * np.sin(theta1) - np.sin(theta1) * np.sin(
                                                   theta3) * np.sin(theta4) + np.cos(theta1) * np.cos(theta2) * np.cos(
                                                   theta3) * np.sin(theta4) + np.cos(theta1) * np.cos(theta2) * np.cos(
                                                   theta4) * np.sin(theta3))) + np.cos(theta1) * np.sin(theta2) * np.sin(
        theta6)) + np.cos(theta7) * (np.sin(theta5) * (
                np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(theta4) * (
                    np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) - np.cos(theta5) * (
                                              np.cos(theta3) * np.cos(theta4) * np.sin(theta1) - np.sin(theta1) * np.sin(theta3) * np.sin(
                                          theta4) + np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.sin(theta4) + np.cos(
                                          theta1) * np.cos(theta2) * np.cos(theta4) * np.sin(theta3)))) * (
                      (37 * np.sin(theta2) * np.sin(theta6) * np.sin(theta7)) / 100 - (
                          27 * np.cos(theta2) * np.cos(theta7) * np.sin(theta5)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta6) * np.sin(theta7)) / 100 - (
                                  47 * np.cos(theta2) * np.cos(theta7) * np.sin(theta6)) / 100 - (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta7) * np.sin(theta5)) / 20 - (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.cos(theta7) * np.sin(theta4)) / 20 - (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta3) * np.sin(theta2) * np.sin(theta6) * np.sin(theta7)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta7) * np.sin(theta5)) / 100 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.cos(theta7) * np.sin(theta4)) / 100 - (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta3)) / 100 - (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2)) / 100 - (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta7)) / 20 + (
                                  37 * np.cos(theta2) * np.cos(theta7) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(theta5) * np.sin(theta7)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta5) * np.sin(theta2) * np.sin(theta4) * np.sin(theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                                  47 * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta7)) / 100 + (
                                  111 * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5)) / 100 + (
                                  111 * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4)) / 100 + (
                                  27 * np.cos(theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(theta6) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta6) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 20 - (
                                  47 * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  27 * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6) * np.sin(theta7)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(
                              theta7)) / 100 - (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(
                              theta2)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta6) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta6) * np.sin(theta3) * np.sin(theta5) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta6) * np.sin(
                              theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.sin(theta2) * np.sin(theta5) * np.sin(
                              theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta4) * np.sin(
                              theta7)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4) * np.sin(
                              theta5)) / 100 + (
                                  111 * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(
                              theta7)) / 100 + (
                                  47 * np.cos(theta4) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(
                              theta5)) / 100 + (
                                  47 * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(
                              theta4)) / 100 - (
                                  37 * np.cos(theta3) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta4) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(theta6) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6) * np.sin(
                              theta7)) / 100 - (
                                  111 * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta7)) / 100) - (np.cos(theta7) * (np.cos(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(theta4) * (
                    np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) + np.sin(theta5) * (
                                                                                           np.cos(theta3) * np.cos(
                                                                                       theta4) * np.sin(theta1) - np.sin(
                                                                                       theta1) * np.sin(theta3) * np.sin(
                                                                                       theta4) + np.cos(theta1) * np.cos(
                                                                                       theta2) * np.cos(theta3) * np.sin(
                                                                                       theta4) + np.cos(theta1) * np.cos(
                                                                                       theta2) * np.cos(theta4) * np.sin(
                                                                                       theta3))) + np.cos(theta1) * np.sin(
        theta2) * np.sin(theta6)) - np.sin(theta7) * (np.sin(theta5) * (
                np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(theta4) * (
                    np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) - np.cos(theta5) * (
                                                            np.cos(theta3) * np.cos(theta4) * np.sin(theta1) - np.sin(theta1) * np.sin(
                                                        theta3) * np.sin(theta4) + np.cos(theta1) * np.cos(theta2) * np.cos(
                                                        theta3) * np.sin(theta4) + np.cos(theta1) * np.cos(theta2) * np.cos(
                                                        theta4) * np.sin(theta3)))) * (
                      (27 * np.cos(theta2) * np.sin(theta5) * np.sin(theta7)) / 20 + (
                          47 * np.cos(theta2) * np.sin(theta6) * np.sin(theta7)) / 100 + (
                                  37 * np.cos(theta7) * np.sin(theta2) * np.sin(theta6)) / 100 - (
                                  37 * np.cos(theta2) * np.cos(theta6) * np.cos(theta7)) / 100 - (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.sin(theta5) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta5) * np.sin(theta4) * np.sin(theta7)) / 20 + (
                                  27 * np.cos(theta3) * np.cos(theta7) * np.sin(theta2) * np.sin(theta6)) / 20 - (
                                  27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7)) / 20 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.sin(theta4) * np.sin(theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3) * np.sin(theta7)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta7)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta5)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4)) / 100 + (
                                  47 * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3)) / 100 + (
                                  27 * np.cos(theta3) * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta6)) / 20 + (
                                  27 * np.cos(theta2) * np.cos(theta6) * np.cos(theta7) * np.sin(theta4) * np.sin(theta5)) / 20 - (
                                  37 * np.cos(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  111 * np.cos(theta3) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  111 * np.cos(theta4) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(theta7)) / 100 - (
                                  111 * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta7)) / 100 - (
                                  47 * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                                  27 * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6)) / 20 - (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(
                              theta7)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta6) * np.cos(theta7) * np.sin(theta4) * np.sin(
                              theta5)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta4) * np.cos(theta6) * np.cos(theta7) * np.sin(theta3) * np.sin(
                              theta5)) / 100 + (
                                  37 * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta3) * np.sin(
                              theta4)) / 100 + (
                                  47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(
                              theta7)) / 100 + (
                                  37 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta6)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta5)) / 100 + (
                                  111 * np.cos(theta3) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta4)) / 100 + (
                                  111 * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(
                              theta3)) / 100 - (
                                  47 * np.cos(theta3) * np.cos(theta6) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta3) * np.cos(theta7) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(
                              theta6)) / 100 - (
                                  47 * np.cos(theta4) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta4) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta5) * np.sin(
                              theta6)) / 100 - (
                                  47 * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta7)) / 100 - (
                                  37 * np.cos(theta5) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta6)) / 100 - (
                                  111 * np.cos(theta6) * np.cos(theta7) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                              theta5)) / 100) - (np.sin(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(theta4) * (
                    np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) + np.sin(theta5) * (
                                                                            np.cos(theta3) * np.cos(theta4) * np.sin(
                                                                        theta1) - np.sin(theta1) * np.sin(theta3) * np.sin(
                                                                        theta4) + np.cos(theta1) * np.cos(theta2) * np.cos(
                                                                        theta3) * np.sin(theta4) + np.cos(theta1) * np.cos(
                                                                        theta2) * np.cos(theta4) * np.sin(theta3))) - np.cos(
        theta1) * np.cos(theta6) * np.sin(theta2)) * ((27 * np.cos(theta6) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4)) / 20 - (
                37 * np.cos(theta6) * np.sin(theta2)) / 100 - (27 * np.cos(theta3) * np.cos(theta6) * np.sin(theta2)) / 20 - (
                                                            27 * np.cos(theta2) * np.cos(theta5) * np.sin(theta6)) / 20 - (
                                                            27 * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.sin(
                                                        theta2)) / 20 - (
                                                            27 * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(
                                                        theta6)) / 20 - (37 * np.cos(theta2) * np.sin(theta6)) / 100 + (
                                                            27 * np.cos(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(
                                                        theta6)) / 20 - (
                                                            37 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(
                                                        theta5) * np.sin(theta6)) / 100 - (
                                                            37 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(
                                                        theta6) * np.sin(theta2)) / 100 + (
                                                            37 * np.cos(theta2) * np.cos(theta3) * np.sin(theta4) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta2) * np.cos(theta4) * np.sin(theta3) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta2) * np.cos(theta5) * np.sin(theta3) * np.sin(
                                                        theta4) * np.sin(theta6)) / 100 + (
                                                            111 * np.cos(theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100 + (
                                                            111 * np.cos(theta3) * np.cos(theta5) * np.sin(theta2) * np.sin(
                                                        theta4) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta3) * np.cos(theta6) * np.sin(theta2) * np.sin(
                                                        theta4) * np.sin(theta5)) / 100 + (
                                                            111 * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(
                                                        theta3) * np.sin(theta6)) / 100 + (
                                                            37 * np.cos(theta4) * np.cos(theta6) * np.sin(theta2) * np.sin(
                                                        theta3) * np.sin(theta5)) / 100 + (
                                                            37 * np.cos(theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(
                                                        theta3) * np.sin(theta4)) / 100 - (
                                                            111 * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
                                                        theta5) * np.sin(theta6)) / 100);
    J32 = (np.cos(theta1) * (
                111 * np.cos(theta2) + 47 * np.cos(theta2) * np.cos(theta6) + 135 * np.sin(theta2) * np.sin(theta3) + 135 * np.cos(
            theta3) * np.sin(theta2) * np.sin(theta4) + 135 * np.cos(theta4) * np.sin(theta2) * np.sin(theta3) + 37 * np.cos(
            theta3) * np.cos(theta4) * np.sin(theta2) * np.sin(theta5) + 37 * np.cos(theta3) * np.cos(theta5) * np.sin(theta2) * np.sin(
            theta4) + 37 * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) - 37 * np.sin(theta2) * np.sin(theta3) * np.sin(
            theta4) * np.sin(theta5) - 47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta2) * np.sin(theta6) + 47 * np.cos(
            theta3) * np.sin(theta2) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6) + 47 * np.cos(theta4) * np.sin(theta2) * np.sin(
            theta3) * np.sin(theta5) * np.sin(theta6) + 47 * np.cos(theta5) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(
            theta6))) / 100;
    J33 = (27 * np.sin(theta1) * np.sin(theta3)) / 20 + (27 * np.cos(theta3) * np.sin(theta1) * np.sin(theta4)) / 20 + (
                27 * np.cos(theta4) * np.sin(theta1) * np.sin(theta3)) / 20 - (
                      27 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) / 20 - (
                      27 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4)) / 20 + (
                      27 * np.cos(theta1) * np.cos(theta2) * np.sin(theta3) * np.sin(theta4)) / 20 + (
                      37 * np.cos(theta3) * np.cos(theta4) * np.sin(theta1) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta3) * np.cos(theta5) * np.sin(theta1) * np.sin(theta4)) / 100 + (
                      37 * np.cos(theta4) * np.cos(theta5) * np.sin(theta1) * np.sin(theta3)) / 100 - (
                      37 * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5)) / 100 + (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta4) * np.sin(theta3) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta5) * np.sin(theta3) * np.sin(theta4)) / 100 - (
                      47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta1) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta3) * np.sin(theta1) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta4) * np.sin(theta1) * np.sin(theta3) * np.sin(theta5) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta5) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta1) * np.cos(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100;
    J34 = (np.sin(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(theta4) * (
                    np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) + np.sin(theta5) * (
                                      np.cos(theta4) * (
                                          np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3)) - np.sin(
                                  theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(
                                  theta3)))) - np.cos(theta1) * np.cos(theta6) * np.sin(theta2)) * (
                      (37 * np.sin(theta6)) / 100 + (27 * np.cos(theta5) * np.sin(theta6)) / 20) + (np.sin(theta7) * (
                np.cos(theta6) * (np.cos(theta5) * (
                    np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(
                theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) + np.sin(theta5) * (
                                           np.cos(theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(
                                       theta3)) - np.sin(theta4) * (
                                                       np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(
                                                   theta3)))) + np.cos(theta1) * np.sin(theta2) * np.sin(theta6)) - np.cos(
        theta7) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3)) - np.sin(theta4) * (
                    np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3))) - np.sin(theta5) * (
                               np.cos(theta4) * (
                                   np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(
                           theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))))) * (
                      (27 * np.cos(theta7) * np.sin(theta5)) / 20 + (47 * np.cos(theta7) * np.sin(theta6)) / 100 + np.sin(theta7) * (
                          (37 * np.cos(theta6)) / 100 + (27 * np.cos(theta5) * np.cos(theta6)) / 20)) - (np.cos(theta7) * (
                np.cos(theta6) * (np.cos(theta5) * (
                    np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(
                theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) + np.sin(theta5) * (
                                           np.cos(theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(
                                       theta3)) - np.sin(theta4) * (
                                                       np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(
                                                   theta3)))) + np.cos(theta1) * np.sin(theta2) * np.sin(theta6)) + np.sin(
        theta7) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3)) - np.sin(theta4) * (
                    np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3))) - np.sin(theta5) * (
                               np.cos(theta4) * (
                                   np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(
                           theta4) * (np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))))) * (
                      (27 * np.sin(theta5) * np.sin(theta7)) / 20 + (47 * np.sin(theta6) * np.sin(theta7)) / 100 - np.cos(theta7) * (
                          (37 * np.cos(theta6)) / 100 + (27 * np.cos(theta5) * np.cos(theta6)) / 20));
    J35 = (37 * np.cos(theta3) * np.cos(theta4) * np.sin(theta1) * np.sin(theta5)) / 100 + (
                37 * np.cos(theta3) * np.cos(theta5) * np.sin(theta1) * np.sin(theta4)) / 100 + (
                      37 * np.cos(theta4) * np.cos(theta5) * np.sin(theta1) * np.sin(theta3)) / 100 - (
                      37 * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5)) / 100 + (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta4) * np.sin(theta3) * np.sin(theta5)) / 100 + (
                      37 * np.cos(theta1) * np.cos(theta2) * np.cos(theta5) * np.sin(theta3) * np.sin(theta4)) / 100 - (
                      47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(theta1) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta3) * np.sin(theta1) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta4) * np.sin(theta1) * np.sin(theta3) * np.sin(theta5) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta5) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta5) * np.sin(theta4) * np.sin(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta1) * np.cos(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5) * np.sin(theta6)) / 100
    J36 = (47 * np.cos(theta6) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                47 * np.cos(theta3) * np.cos(theta4) * np.cos(theta6) * np.sin(theta1) * np.sin(theta5)) / 100 - (
                      47 * np.cos(theta3) * np.cos(theta5) * np.cos(theta6) * np.sin(theta1) * np.sin(theta4)) / 100 - (
                      47 * np.cos(theta4) * np.cos(theta5) * np.cos(theta6) * np.sin(theta1) * np.sin(theta3)) / 100 - (
                      47 * np.cos(theta1) * np.sin(theta2) * np.sin(theta6)) / 100 + (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.cos(theta6)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta6) * np.sin(theta4) * np.sin(theta5)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta4) * np.cos(theta6) * np.sin(theta3) * np.sin(theta5)) / 100 - (
                      47 * np.cos(theta1) * np.cos(theta2) * np.cos(theta5) * np.cos(theta6) * np.sin(theta3) * np.sin(theta4)) / 100
    J37 = 0
    J41 = 0
    J42 = np.cos(theta1)
    J43 = np.sin(theta1) * np.sin(theta2)
    J44 = np.sin(theta1) * np.sin(theta2)
    J45 = np.sin(theta1) * np.sin(theta2)
    J46 = np.cos(theta1) * np.cos(theta3) * np.cos(theta4) * np.cos(theta5) - np.cos(theta1) * np.cos(theta3) * np.sin(theta4) * np.sin(
        theta5) - np.cos(theta1) * np.cos(theta4) * np.sin(theta3) * np.sin(theta5) - np.cos(theta1) * np.cos(theta5) * np.sin(theta3) * np.sin(
        theta4) - np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta1) * np.sin(theta5) - np.cos(theta2) * np.cos(theta3) * np.cos(
        theta5) * np.sin(theta1) * np.sin(theta4) - np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta1) * np.sin(theta3) + np.cos(
        theta2) * np.sin(theta1) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)
    J47 = np.sin(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(theta1)) + np.sin(theta4) * (
                    np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3))) + np.sin(theta5) * (
                                     np.cos(theta4) * (
                                         np.cos(theta1) * np.cos(theta3) - np.cos(theta2) * np.sin(theta1) * np.sin(theta3)) - np.sin(
                                 theta4) * (np.cos(theta1) * np.sin(theta3) + np.cos(theta2) * np.cos(theta3) * np.sin(
                                 theta1)))) + np.cos(theta6) * np.sin(theta1) * np.sin(theta2)
    J51 = (np.cos(theta2) * np.cos(theta6) - np.cos(theta3 + theta4 + theta5) * np.sin(theta2) * np.sin(theta6)) ** 2 + (
                np.cos(theta2) * np.cos(theta7) * np.sin(theta6) - np.sin(theta3 + theta4 + theta5) * np.sin(theta2) * np.sin(
            theta7) + np.cos(theta3 + theta4 + theta5) * np.cos(theta6) * np.cos(theta7) * np.sin(theta2)) ** 2 + (
                      np.cos(theta2) * np.sin(theta6) * np.sin(theta7) + np.sin(theta3 + theta4 + theta5) * np.cos(theta7) * np.sin(
                  theta2) + np.cos(theta3 + theta4 + theta5) * np.cos(theta6) * np.sin(theta2) * np.sin(theta7)) ** 2
    J52 = 0
    J53 = np.cos(theta2)
    J54 = np.cos(theta2)
    J55 = np.cos(theta2)
    J56 = np.cos(theta3 - theta2 + theta4 + theta5) / 2 - np.cos(theta2 + theta3 + theta4 + theta5) / 2
    J57 = np.cos(theta2) * np.cos(theta6) - np.cos(theta3 + theta4 + theta5) * np.sin(theta2) * np.sin(theta6)
    J61 = 0
    J62 = -np.sin(theta1)
    J63 = np.cos(theta1) * np.sin(theta2)
    J64 = np.cos(theta1) * np.sin(theta2)
    J65 = np.cos(theta1) * np.sin(theta2)
    J66 = np.cos(theta3) * np.sin(theta1) * np.sin(theta4) * np.sin(theta5) - np.cos(theta3) * np.cos(theta4) * np.cos(theta5) * np.sin(
        theta1) + np.cos(theta4) * np.sin(theta1) * np.sin(theta3) * np.sin(theta5) + np.cos(theta5) * np.sin(theta1) * np.sin(theta3) * np.sin(
        theta4) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) * np.sin(theta5) - np.cos(theta1) * np.cos(theta2) * np.cos(
        theta3) * np.cos(theta5) * np.sin(theta4) - np.cos(theta1) * np.cos(theta2) * np.cos(theta4) * np.cos(theta5) * np.sin(theta3) + np.cos(
        theta1) * np.cos(theta2) * np.sin(theta3) * np.sin(theta4) * np.sin(theta5)
    J67 = np.cos(theta1) * np.cos(theta6) * np.sin(theta2) - np.sin(theta6) * (np.cos(theta5) * (
                np.cos(theta4) * (np.sin(theta1) * np.sin(theta3) - np.cos(theta1) * np.cos(theta2) * np.cos(theta3)) + np.sin(theta4) * (
                    np.cos(theta3) * np.sin(theta1) + np.cos(theta1) * np.cos(theta2) * np.sin(theta3))) + np.sin(theta5) * (
                                                                               np.cos(theta4) * (
                                                                                   np.cos(theta3) * np.sin(theta1) + np.cos(
                                                                               theta1) * np.cos(theta2) * np.sin(
                                                                               theta3)) - np.sin(theta4) * (
                                                                                           np.sin(theta1) * np.sin(
                                                                                       theta3) - np.cos(theta1) * np.cos(
                                                                                       theta2) * np.cos(theta3))))

    J = np.array([[J11, J12, J13, J14, J15, J16, J17],
    [J21, J22, J23, J24, J25, J26, J27],
    [J31, J32, J33, J34, J35, J36, J37],
    [J41, J42, J43, J44, J45, J46, J47],
    [J51, J52, J53, J54, J55, J56, J57],
    [J61, J62, J63, J64, J65, J66, J67]])

    J13 = np.array([[J11, J12, J13, J14, J15, J16, J17],
    [J21, J22, J23, J24, J25, J26, J27],
    [J31, J32, J33, J34, J35, J36, J37]])

    J46 = np.array([[J41, J42, J43, J44, J45, J46, J47],
    [J51, J52, J53, J54, J55, J56, J57],
    [J61, J62, J63, J64, J65, J66, J67]])

    delta_u_initial = np.array([random.uniform(-1.5, 3.5), random.uniform(-1, 1), random.uniform(0, 2),random.uniform(-np.pi/2,np.pi/2),random.uniform(-np.pi/2,np.pi/2),random.uniform(-np.pi/2,np.pi/2)])

    delta_u = delta_u_initial / np.linalg.norm(delta_u_initial) * 0.01
    omega = np.dot(np.array([[0, -np.sin(delta_u[3]), np.cos(delta_u[3]) * np.cos(delta_u[4])],
    [0, np.cos(delta_u[5]), np.sin(delta_u[3]) * np.cos(delta_u[4])],
    [1, 0, -np.sin(delta_u[4])]]), [[delta_u[3]],[delta_u[4]], [delta_u[5]]])

    # 输出动作（其实是末端运动目标速度）
    delta_error = [delta_u[0], delta_u[1], delta_u[2], omega[0], omega[1], omega[2]]

    # 期望的关节运动角速度
    del_q = np.dot(np.linalg.pinv(J), delta_error)

    input_data1 = np.array([theta1, theta2, theta3,theta4,theta5,theta6,theta7, delta_u[0], delta_u[1], delta_u[2], delta_u[3], delta_u[4], delta_u[5]])
    output_data1 = np.array([del_q[0],del_q[1],del_q[2],del_q[3],del_q[4],del_q[5],del_q[6]])
    total_data1 = np.array([theta1, theta2, theta3,theta4,theta5,theta6,theta7, delta_u[0], delta_u[1], delta_u[2], delta_u[3], delta_u[4], delta_u[5],\
                            del_q[0][0],del_q[1][0],del_q[2][0],del_q[3][0],del_q[4][0],del_q[5][0],del_q[6][0]])
    input_data.append(input_data1)
    output_data.append(output_data1)
    total_data.append(total_data1)
    if i % 1000 == 999:
        print(i)

np.save("input_data", input_data)
np.save("output_data", output_data)
np.save("total_data", total_data)

np.load("input_data.npy")
np.load("output_data.npy")
print(np.load("total_data.npy"))


