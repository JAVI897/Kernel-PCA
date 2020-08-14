#!/usr/bin/env python
# coding: utf-8

import numpy as np

class kernel:
    
    def __init__(self, gamma = 1, sigma = 1, d_anova = 1, d_poly = 2, d_power = 1, alpha = 1, c = 0):
        self.gamma = gamma
        self.sigma = sigma
        self.d_anova = d_anova
        self.alpha = alpha
        self.c = c
        self.d_poly = d_poly
        self.d_power = d_power
        
    def linear(self, x, y):
        """
        k(x, y) = <x, y> + c
        Hiperparámetros: c
        """
        return x.T@y + self.c
    
    def rbf(self, x, y):
        """
        k(x, y) = exp(- gamma * ||x-y||^2)
        Hiperparámetros: gamma
        """
        return np.exp(- self.gamma * (np.linalg.norm(x-y)**2))
    
    def exp(self, x, y):
        """
        k(x, y) = exp(- ||x-y|| / (2 * sigma^2) )
        Hiperparámetros: sigma
        """
        return np.exp(- (1/ (2*self.sigma**2)) * np.linalg.norm(x-y))
    
    def laplacian(self, x, y):
        """
        k(x, y) = exp(- ||x-y|| / sigma )
        Hiperparámetros: sigma
        """
        return np.exp(- (1/self.sigma) * np.linalg.norm(x-y))
    
    def anova(self, x, y):
        """
        k(x, y) = sum( exp(- sigma * ((x_i - y_i)^2))^d_anova )
        Hiperparámetros: sigma, d_anova
        """
        suma = 0
        for i in range(0, len(x)):
            term_1 = - self.sigma * ( (x[i] - y[i] )**2 )
            suma += np.exp(term_1) ** self.d_anova
        return suma
    
    def polynomial(self, x, y):
        """
        k(x, y) = (alpha * <x, y> + c)^d
        Hiperparámetros: alpha, c, d_poly
        """
        return (self.alpha * (x.T@y) + self.c)**self.d_poly
    
    def sigmoid(self, x, y):
        """
        k(x, y) = tanh( alpha * <x, y> + c)
        Hiperparámetros: alpha, c
        """
        return np.tanh(self.alpha * (x.T@y) + self.c)
    
    def rotational_quadratic(self, x, y):
        """
        k(x, y) = 1 - (||x-y||^2 / ||x-y||^2 + c)
        Hiperparámetros: c
        """
        dist = np.linalg.norm(x-y)
        return 1 - (dist**2 / (dist**2 + self.c))
    
    def multiquadric(self, x, y):
        """
        k(x, y) = sqrt(||x-y||^2 + c^2)
        Hiperparámetros: c
        """
        return np.sqrt(np.linalg.norm(x-y)**2 + self.c**2)
    
    def power(self, x, y):
        """
        k(x, y) = -||x-y||^d
        Hiperparámetros: d_power
        """
        return - np.linalg.norm(x-y)**self.d_power
    
    def spherical(self, x, y):
        dist = np.linalg.norm(x-y)
        if dist > self.sigma:
            return 0
        return 1 - (3/2)*(dist/self.sigma)+(1/2)*((dist/self.sigma)**3)
    
    def circular(self, x, y):
        dist = np.linalg.norm(x-y)
        if dist > self.sigma:
            return 0
        return (2/np.pi)*np.arccos(- dist/self.sigma)-(2/np.pi)*(dist/self.sigma)*np.sqrt(1 - (dist/self.sigma)**2)