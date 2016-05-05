from astropy.io import ascii
import requests
from acispy.utils import get_time, calc_off_nom_rolls, state_units
import numpy as np
from Chandra.cmd_states import fetch_states
import astropy.units as apu
from acispy.data_collection import DataCollection

class States(DataCollection):

    def __init__(self, table):
        self.table = {}
        for k, v in table.items():
            if k in state_units:
                self.table[k] = v*getattr(apu, state_units[k])
            else:
                self.table[k] = v
        if set(["q1","q2","q3","q4"]) < set(self.table.keys()):
            self.table["off_nominal_roll"] = calc_off_nom_rolls(table)*apu.deg
        self.times = np.append(table["tstart"], table["tstop"][-1])*apu.s

    @classmethod
    def from_database(cls, states, tstart, tstop):
        st = states[:]
        if "off_nominal_roll" in states:
            st.remove("off_nominal_roll")
            st += ["q1", "q2", "q3", "q4"]
        t = fetch_states(tstart, tstop, vals=st)
        table = dict((k, t[k]) for k in t.dtype.names)
        return cls(table)

    @classmethod
    def from_load(cls, load):
        url = "http://cxc.cfa.harvard.edu/acis/DPA_thermPredic/"
        url += "%s/ofls%s/states.dat" % (load[:-1].upper(), load[-1].lower())
        u = requests.get(url)
        t = ascii.read(u.text)
        table = dict((k, t[k].data) for k in t.keys())
        return cls(table)

    def get_states(self, time):
        time = get_time(time).secs
        # We have this if we need it
        err = "The time %s is not within the selected time frame!" % time
        if time < self["tstart"][0].value:
            raise RuntimeError(err)
        idx = np.searchsorted(self["tstart"].value, time)-1
        try:
            self["tstart"][idx].value
        except IndexError:
            raise RuntimeError(err)
        state = {}
        for key in self.keys():
            state[key] = self[key][idx]
        return state

    @property
    def current_states(self):
        return self.get_states("now")
