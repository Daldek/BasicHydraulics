import matplotlib.pyplot as plt
import math
from fractions import Fraction

class Channel():
    '''Simple class to define channel's XS'''

    def __init__(self, inclination, roughness_n):
        '''Basic input parameters'''
        self.inclination = inclination
        self.roughness_n = roughness_n
        self._hydraulic_radius = None
        self._velocity = None
        self._flow_rate = None

    def hydraulic_radius(self):
        if self._hydraulic_radius is None:
            self._hydraulic_radius = self.cross_sectional_area()/self.wetted_circut()
        return self._hydraulic_radius

    def velocity(self):
        if self._velocity is None:
            self._velocity = (1/self.roughness_n) *\
            math.pow(self.hydraulic_radius(), Fraction(2, 3)) *\
            math.pow(self.inclination, 0.5)
        return self._velocity

    def flow_rate(self):
        if self._flow_rate is None:
            self._flow_rate = self.velocity()*self.cross_sectional_area()
        return self._flow_rate

class TrapezoidalChannel(Channel):
    '''Class for trapezoidal channels'''

    def __init__(self, depth, width, bank_inclination_m, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.width = width
        self.bank_inclination_m = bank_inclination_m
        self._cross_sectional_area = None
        self._wetted_circut = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = ((self.width+((2*(self.depth*self.bank_inclination_m))+self.width))/2.0)*self.depth
        return self._cross_sectional_area

    def wetted_circut(self):
        if self._wetted_circut is None:
            xx = math.pow((self.bank_inclination_m*self.depth), 2)+math.pow(self.depth, 2)
            x = math.sqrt(xx)
            self._wetted_circut = (2*x)+self.width
        return self._wetted_circut
    
    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, (self.depth*100) + 1, 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_circut = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            flow_rates[self.depth] = round(self.flow_rate(), 2)
        return flow_rates

class SemiCircularChannel(Channel):
    '''Class for circular channels'''

    def __init__(self, radius, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.radius = radius
        self._cross_sectional_area = None
        self._wetted_circut = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = math.pi * math.pow(self.radius, 2) * 0.5
        return self._cross_sectional_area

    def wetted_circut(self):
        if self._wetted_circut is None:
            self._wetted_circut = math.pi * self.radius
        return self._wetted_circut    
