"""
This file contains the ToR-ORd model, to be used in inference.
"""
from numpy import exp, log, sqrt

#  #      Cardiac model ToR-ORd
#  #      Copyright (C) 2019 Jakub Tomek. Contact: jakub.tomek.mff@gmail.com
#  #
#  #      This program is free software: you can redistribute it and/or modify
#  #      it under the terms of the GNU General Public License as published by
#  #      the Free Software Foundation, either version 3 of the License, or
#  #      (at your option) any later version.
#  #
#  #      This program is distributed in the hope that it will be useful,
#  #      but WITHOUT ANY WARRANTY without even the implied warranty of
#  #      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  #      GNU General Public License for more details.
#  #
#  #      You should have received a copy of the GNU General Public License
#  #      along with this program.  If not, see <https://www.gnu.org/licenses/>.


class ToRORdRHS:

    def __init__(self, t, y, G_Na, P_Cab, G_Kr, G_ClCa, G_Clb, stim_duration,
                 stim_amplitude, multipliers, concs_and_fractions, celltype=0,
                 flag_ode=True):

        # Initial parameters
        self.t = t
        self.G_Na = G_Na
        self.P_Cab = P_Cab
        self.G_Kr = G_Kr
        self.G_ClCa = G_ClCa
        self.G_Clb = G_Clb
        self.stim_duration = stim_duration
        self.stim_amplitude = stim_amplitude
        self.multipliers = multipliers
        self.concs_and_fractions = concs_and_fractions
        self.nao = concs_and_fractions["Na_o"]
        self.cao = concs_and_fractions["Ca_o"]
        self.ko = concs_and_fractions["K_o"]
        self.celltype = celltype
        self.flag_ode = flag_ode

        # physical constants
        self.R = 8314.0
        self.T = 310.0
        self.F = 96485.0

        # cell geometry
        L = 0.01
        rad = 0.0011
        vcell = 1000 * 3.14 * rad * rad * L
        Ageo = 2 * 3.14 * rad * rad + 2 * 3.14 * rad * L
        self.Acap = 2 * Ageo
        self.vmyo = 0.68 * vcell
        self.vnsr = 0.0552 * vcell
        self.vjsr = 0.0048 * vcell
        self.vss = 0.02 * vcell

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # membrane potential
        self.v = y[0]

        # ionic compartment concentrations
        self.nai = y[1]
        self.nass = y[2]
        self.ki = y[3]
        self.kss = y[4]
        self.cai = y[5]
        self.cass = y[6]
        self.cansr = y[7]
        self.cajsr = y[8]

        # I_Na gating variables
        self.m = y[9]
        self.hp = y[10]
        self.h = y[11]
        self.j = y[12]
        self.jp = y[13]

        # I_NaL gating variables
        self.mL = y[14]
        self.hL = y[15]
        self.hLp = y[16]

        # I_to gating variables
        self.a = y[17]
        self.iF = y[18]
        self.iS = y[19]
        self.ap = y[20]
        self.iFp = y[21]
        self.iSp = y[22]

        # I_CaL gating variables
        self.d = y[23]
        self.ff = y[24]
        self.fs = y[25]
        self.fcaf = y[26]
        self.fcas = y[27]
        self.jca = y[28]
        self.nca = y[29]
        self.nca_i = y[30]
        self.ffp = y[31]
        self.fcafp = y[32]

        # I_Ks gating variables
        self.xs1 = y[33]
        self.xs2 = y[34]

        # J_rel, CaMKII and I_Kr variables
        self.Jrel_np = y[35]
        self.CaMKt = y[36]
        self.ikr_c0 = y[37]
        self.ikr_c1 = y[38]
        self.ikr_c2 = y[39]
        self.ikr_o = y[40]
        self.ikr_i = y[41]
        self.Jrel_p = y[42]

        self.cli = 24  # Intracellular Cl  [mM]
        self.clo = 150  # Extracellular Cl  [mM]
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # CaMK constants
        self.KmCaMK = 0.15

        aCaMK = 0.05
        bCaMK = 0.00068
        CaMKo = 0.05
        KmCaM = 0.0015
        # update CaMK
        CaMKb = CaMKo * (1.0 - self.CaMKt) / (1.0 + KmCaM / self.cass)
        self.CaMKa = CaMKb + self.CaMKt
        self.dCaMKt = aCaMK * CaMKb * (CaMKb + self.CaMKt) - bCaMK * self.CaMKt

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # reversal potentials
        self.ENa = (self.R * self.T / self.F) * log(self.nao / self.nai)
        self.EK = (self.R * self.T / self.F) * log(self.ko / self.ki)
        PKNa = 0.01833
        self.EKs = (self.R * self.T / self.F) * log((self.ko + PKNa * self.nao) / (self.ki + PKNa * self.nai))

        # convenient shorthand calculations
        self.vffrt = self.v * self.F * self.F / (self.R * self.T)
        self.vfrt = self.v * self.F / (self.R * self.T)
        self.frt = self.F / (self.R * self.T)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.fINap = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))
        self.fINaLp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))
        self.fItop = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))
        self.fICaLp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))

    def evaluate(self):
        # #  INa
        INa, dm, dh, dhp, dj, djp = self.getINa_Grandi()

        # #  INaL
        INaL, dmL, dhL, dhLp = self.getINaL_ORd2011()

        # #  ITo
        Ito, da, diF, diS, dap, diFp, diSp = self.getITo_ORd2011()

        # #  ICaL
        (ICaL_ss, ICaNa_ss, ICaK_ss, ICaL_i, ICaNa_i, ICaK_i, dd, dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i, dffp,
         dfcafp, PhiCaL_ss, PhiCaL_i, gammaCaoMyo, gammaCaiMyo) = self.getICaL_ORd2011_jt()

        ICaL = ICaL_ss + ICaL_i
        ICaNa = ICaNa_ss + ICaNa_i
        ICaK = ICaK_ss + ICaK_i
        ICaL_tot = ICaL + ICaNa + ICaK
        # #  IKr
        IKr, dt_ikr_c0, dt_ikr_c1, dt_ikr_c2, dt_ikr_o, dt_ikr_i = self.getIKr_ORd2011_MM()

        # #  IKs
        IKs, dxs1, dxs2 = self.getIKs_ORd2011()

        # #  IK1
        IK1 = self.getIK1_CRLP()

        # #  INaCa
        INaCa_i, INaCa_ss = self.getINaCa_ORd2011()

        # #  INaK
        INaK = self.getINaK_ORd2011()

        # #  Minor/background currents
        # calculate IKb
        xkb = 1.0 / (1.0 + exp(-(self.v - 10.8968) / 23.9871))
        GKb = 0.0189 * self.multipliers["I_Kb"]
        if self.celltype == 1:
            GKb = GKb * 0.6
        IKb = GKb * xkb * (self.v - self.EK)

        # calculate INab
        PNab = 1.9239e-09 * self.multipliers["I_Nab"]
        INab = PNab * self.vffrt * (self.nai * exp(self.vfrt) - self.nao) / (exp(self.vfrt) - 1.0)

        # calculate ICab
        PCab = 5.9194e-08 * self.multipliers["I_Cab"]
        #
        ICab = PCab * 4.0 * self.vffrt * (gammaCaiMyo * self.cai * exp(2.0 * self.vfrt) - gammaCaoMyo * self.cao) / (
                exp(2.0 * self.vfrt) - 1.0)

        # calculate IpCa
        GpCa = 5e-04 * self.multipliers["I_pCa"]
        IpCa = GpCa * self.cai / (0.0005 + self.cai)

        # #  Chloride
        #  I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current

        ecl = (self.R * self.T / self.F) * log(self.cli / self.clo)  # [mV]

        Fjunc = 1
        Fsl = 1 - Fjunc  # fraction in SS and in myoplasm - as per literature, I(Ca)Cl is in junctional subspace

        # Both of the below conductances are some of the inferred parameters
        GClCa = self.multipliers["I_CaCl"] * self.G_ClCa  # [mS/uF]
        GClB = self.multipliers["I_Clb"] * self.G_Clb  # [mS/uF] #
        KdClCa = 0.1  # [mM]

        I_ClCa_junc = Fjunc * GClCa / (1 + KdClCa / self.cass) * (self.v - ecl)
        I_ClCa_sl = Fsl * GClCa / (1 + KdClCa / self.cai) * (self.v - ecl)

        I_ClCa = I_ClCa_junc + I_ClCa_sl
        I_Clbk = GClB * (self.v - ecl)

        # #  Calcium handling
        # calculate ryanodione receptor calcium induced calcium release from the jsr

        # #  Jrel
        fJrelp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))
        Jrel, dJrel_np, dJrel_p = self.getJrel_ORd2011(ICaL, fJrelp)

        fJupp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))
        Jup, Jleak = self.getJup_ORd2011(fJupp)

        # calculate translocation flux
        Jtr = (self.cansr - self.cajsr) / 60

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # #  calculate the stimulus current, Istim
        amp = self.stim_amplitude
        duration = self.stim_duration
        if self.t <= duration:
            Istim = amp
        else:
            Istim = 0.0

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # update the membrane voltage

        dv = -(INa + INaL + Ito + ICaL + ICaNa + ICaK + IKr + IKs + IK1 + INaCa_i + INaCa_ss + INaK + INab + IKb +
               IpCa + ICab + I_ClCa + I_Clbk + Istim)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # calculate diffusion fluxes
        JdiffNa = (self.nass - self.nai) / 2.0
        JdiffK = (self.kss - self.ki) / 2.0
        Jdiff = (self.cass - self.cai) / 0.2

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # calcium buffer constants
        cmdnmax = 0.05
        if self.celltype == 1:
            cmdnmax = cmdnmax * 1.3
        kmcmdn = 0.00238
        trpnmax = 0.07
        kmtrpn = 0.0005
        BSRmax = 0.047
        KmBSR = 0.00087
        BSLmax = 1.124
        KmBSL = 0.0087
        csqnmax = 10.0
        kmcsqn = 0.8

        # update intracellular concentrations, using buffers for cai, cass, cajsr
        dnai = -(ICaNa_i + INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + INab) * self.Acap / (
                self.F * self.vmyo) + JdiffNa * self.vss / self.vmyo
        dnass = -(ICaNa_ss + 3.0 * INaCa_ss) * self.Acap / (self.F * self.vss) - JdiffNa

        dki = -(ICaK_i + Ito + IKr + IKs + IK1 + IKb + Istim - 2.0 * INaK) * self.Acap / (
                self.F * self.vmyo) + JdiffK * self.vss / self.vmyo
        dkss = -ICaK_ss * self.Acap / (self.F * self.vss) - JdiffK

        Bcai = (1.0 / (1.0 + cmdnmax * kmcmdn / (kmcmdn + self.cai) ** 2.0
                       + trpnmax * kmtrpn / (kmtrpn + self.cai) ** 2.0))
        dcai = Bcai * (-(ICaL_i + IpCa + ICab - 2.0 * INaCa_i) * self.Acap / (
                2.0 * self.F * self.vmyo) - Jup * self.vnsr / self.vmyo + Jdiff * self.vss / self.vmyo)

        Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + self.cass) ** 2.0 + BSLmax * KmBSL / (KmBSL + self.cass) ** 2.0)
        dcass = Bcass * (-(ICaL_ss - 2.0 * INaCa_ss) * self.Acap / (
                2.0 * self.F * self.vss) + Jrel * self.vjsr / self.vss - Jdiff)

        dcansr = Jup - Jtr * self.vjsr / self.vnsr

        Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + self.cajsr) ** 2.0)
        dcajsr = Bcajsr * (Jtr - Jrel)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # output the state vector when ode_flag==1, and the calculated currents and fluxes otherwise
        if self.flag_ode == 1:
            return [dv, dnai, dnass, dki, dkss, dcai, dcass, dcansr, dcajsr, dm, dhp,
                    dh, dj, djp, dmL, dhL, dhLp, da, diF, diS, dap, diFp, diSp, dd,
                    dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i, dffp, dfcafp, dxs1,
                    dxs2, dJrel_np, self.dCaMKt, dt_ikr_c0, dt_ikr_c1, dt_ikr_c2, dt_ikr_o,
                    dt_ikr_i, dJrel_p]
        else:
            return [INa, INaL, Ito, ICaL, IKr, IKs, IK1, INaCa_i, INaCa_ss, INaK,
                    IKb, INab, ICab, IpCa, Jdiff, JdiffNa, JdiffK, Jup, Jleak, Jtr,
                    Jrel, self.CaMKa, Istim, self.fINap, self.fINaLp, self.fICaLp,
                    fJrelp, fJupp, self.cajsr, self.cansr, PhiCaL_ss, self.v,
                    ICaL_i, I_ClCa, I_Clbk, ICaL_tot]

    def getINa_Grandi(self):
        """INa formulations"""

        #  The Grandi implementation updated with INa phosphorylation.
        # #  m gate
        mss = 1 / ((1 + exp(-(56.86 + self.v) / 9.03)) ** 2)
        taum = 0.1292 * exp(-((self.v + 45.79) / 15.54) ** 2) + 0.06487 * exp(-((self.v - 4.823) / 51.12) ** 2)
        dm = (mss - self.m) / taum

        # #  h gate
        ah = (self.v >= -40) * 0 + (self.v < -40) * (0.057 * exp(-(self.v + 80) / 6.8))
        bh = ((self.v >= -40) * (0.77 / (0.13 * (1 + exp(-(self.v + 10.66) / 11.1))))
              + (self.v < -40) * (2.7 * exp(0.079 * self.v) + 3.1 * 10 ** 5 * exp(0.3485 * self.v)))
        tauh = 1 / (ah + bh)
        hss = 1 / ((1 + exp((self.v + 71.55) / 7.43)) ** 2)
        dh = (hss - self.h) / tauh
        # #  j gate
        aj = ((self.v >= -40) * 0
              + (self.v < -40) * (((-2.5428 * 10 ** 4 * exp(0.2444 * self.v)
                                    - 6.948 * 10 ** -6 * exp(-0.04391 * self.v))
                                   * (self.v + 37.78)) / (1 + exp(0.311 * (self.v + 79.23)))))
        bj = ((self.v >= -40) * ((0.6 * exp(0.057 * self.v)) / (1 + exp(-0.1 * (self.v + 32))))
              + (self.v < -40) * ((0.02424 * exp(-0.01052 * self.v)) / (1 + exp(-0.1378 * (self.v + 40.14)))))
        tauj = 1 / (aj + bj)
        jss = 1 / ((1 + exp((self.v + 71.55) / 7.43)) ** 2)
        dj = (jss - self.j) / tauj

        # #  h phosphorylated
        hssp = 1 / ((1 + exp((self.v + 71.55 + 6) / 7.43)) ** 2)
        dhp = (hssp - self.hp) / tauh
        # #  j phosphorylated
        taujp = 1.46 * tauj
        djp = (jss - self.jp) / taujp

        # Below is using one of our potentially inferred parameters
        GNa = self.G_Na
        INa = self.multipliers["I_Na"] * GNa * (self.v - self.ENa) * self.m ** 3.0 * (
                (1.0 - self.fINap) * self.h * self.j + self.fINap * self.hp * self.jp)

        return INa, dm, dh, dhp, dj, djp

    def getINaL_ORd2011(self):
        """INaL formulations"""
        # calculate INaL
        mLss = 1.0 / (1.0 + exp((-(self.v + 42.85)) / 5.264))
        tm = 0.1292 * exp(-((self.v + 45.79) / 15.54) ** 2) + 0.06487 * exp(-((self.v - 4.823) / 51.12) ** 2)
        tmL = tm
        dmL = (mLss - self.mL) / tmL
        hLss = 1.0 / (1.0 + exp((self.v + 87.61) / 7.488))
        thL = 200.0
        dhL = (hLss - self.hL) / thL
        hLssp = 1.0 / (1.0 + exp((self.v + 93.81) / 7.488))
        thLp = 3.0 * thL
        dhLp = (hLssp - self.hLp) / thLp
        GNaL = 0.0279 * self.multipliers["I_NaL"]
        if self.celltype == 1:
            GNaL = GNaL * 0.6

        INaL = GNaL * (self.v - self.ENa) * self.mL * ((1.0 - self.fINaLp) * self.hL + self.fINaLp * self.hLp)
        return INaL, dmL, dhL, dhLp

    def getITo_ORd2011(self):
        """Ito formulations"""
        # calculate Ito
        ass = 1.0 / (1.0 + exp((-(self.v - 14.34)) / 14.82))
        ta = 1.0515 / (1.0 / (1.2089 * (1.0 + exp(-(self.v - 18.4099) / 29.3814))) + 3.5 / (
                1.0 + exp((self.v + 100.0) / 29.3814)))
        da = (ass - self.a) / ta
        iss = 1.0 / (1.0 + exp((self.v + 43.94) / 5.711))
        if self.celltype == 1:
            delta_epi = 1.0 - (0.95 / (1.0 + exp((self.v + 70.0) / 5.0)))
        else:
            delta_epi = 1.0
        tiF = 4.562 + 1 / (0.3933 * exp((-(self.v + 100.0)) / 100.0) + 0.08004 * exp((self.v + 50.0) / 16.59))
        tiS = 23.62 + 1 / (0.001416 * exp((-(self.v + 96.52)) / 59.05) + 1.780e-8 * exp((self.v + 114.1) / 8.079))
        tiF = tiF * delta_epi
        tiS = tiS * delta_epi
        AiF = 1.0 / (1.0 + exp((self.v - 213.6) / 151.2))
        AiS = 1.0 - AiF
        diF = (iss - self.iF) / tiF
        diS = (iss - self.iS) / tiS
        i = AiF * self.iF + AiS * self.iS
        assp = 1.0 / (1.0 + exp((-(self.v - 24.34)) / 14.82))
        dap = (assp - self.ap) / ta
        dti_develop = 1.354 + 1.0e-4 / (exp((self.v - 167.4) / 15.89) + exp(-(self.v - 12.23) / 0.2154))
        dti_recover = 1.0 - 0.5 / (1.0 + exp((self.v + 70.0) / 20.0))
        tiFp = dti_develop * dti_recover * tiF
        tiSp = dti_develop * dti_recover * tiS
        diFp = (iss - self.iFp) / tiFp
        diSp = (iss - self.iSp) / tiSp
        ip = AiF * self.iFp + AiS * self.iSp
        Gto = 0.16 * self.multipliers["I_to"]
        if self.celltype == 1:
            Gto = Gto * 2.0
        elif self.celltype == 2:
            Gto = Gto * 2.0

        Ito = Gto * (self.v - self.EK) * ((1.0 - self.fItop) * self.a * i + self.fItop * self.ap * ip)

        return Ito, da, diF, diS, dap, diFp, diSp

    #  a variant updated by jakub, using a changed activation curve
    #  it computes both ICaL in subspace and myoplasm (_i)
    def getICaL_ORd2011_jt(self):
        """a variant updated by jakub, using a changed activation curve -
        it computes both ICaL in subspace and myoplasm (_i)"""

        # calculate ICaL, ICaNa, ICaK

        dss = 1.0763 * exp(-1.0070 * exp(-0.0829 * self.v))  # magyar
        if self.v > 31.4978:  # activation cannot be greater than 1
            dss = 1

        td = 0.6 + 1.0 / (exp(-0.05 * (self.v + 6.0)) + exp(0.09 * (self.v + 14.0)))

        dd = (dss - self.d) / td
        fss = 1.0 / (1.0 + exp((self.v + 19.58) / 3.696))
        tff = 7.0 + 1.0 / (0.0045 * exp(-(self.v + 20.0) / 10.0) + 0.0045 * exp((self.v + 20.0) / 10.0))
        tfs = 1000.0 + 1.0 / (0.000035 * exp(-(self.v + 5.0) / 4.0) + 0.000035 * exp((self.v + 5.0) / 6.0))
        Aff = 0.6
        Afs = 1.0 - Aff
        dff = (fss - self.ff) / tff
        dfs = (fss - self.fs) / tfs
        f = Aff * self.ff + Afs * self.fs
        fcass = fss
        tfcaf = 7.0 + 1.0 / (0.04 * exp(-(self.v - 4.0) / 7.0) + 0.04 * exp((self.v - 4.0) / 7.0))
        tfcas = 100.0 + 1.0 / (0.00012 * exp(-self.v / 3.0) + 0.00012 * exp(self.v / 7.0))

        Afcaf = 0.3 + 0.6 / (1.0 + exp((self.v - 10.0) / 10.0))

        Afcas = 1.0 - Afcaf
        dfcaf = (fcass - self.fcaf) / tfcaf
        dfcas = (fcass - self.fcas) / tfcas
        fca = Afcaf * self.fcaf + Afcas * self.fcas

        tjca = 75
        jcass = 1.0 / (1.0 + exp((self.v + 18.08) / 2.7916))
        djca = (jcass - self.jca) / tjca
        tffp = 2.5 * tff
        dffp = (fss - self.ffp) / tffp
        fp = Aff * self.ffp + Afs * self.fs
        tfcafp = 2.5 * tfcaf
        dfcafp = (fcass - self.fcafp) / tfcafp
        fcap = Afcaf * self.fcafp + Afcas * self.fcas

        # #  SS nca
        Kmn = 0.002
        k2n = 500.0
        km2n = self.jca * 1
        anca = 1.0 / (k2n / km2n + (1.0 + Kmn / self.cass) ** 4.0)
        dnca = anca * k2n - self.nca * km2n

        # #  myoplasmic nca
        anca_i = 1.0 / (k2n / km2n + (1.0 + Kmn / self.cai) ** 4.0)
        dnca_i = anca_i * k2n - self.nca_i * km2n

        # #  SS driving force
        clo = 150
        cli = 24
        # ionic strength outside. /1000 is for things being in micromolar
        Io = 0.5 * (self.nao + self.ko + clo + 4 * self.cao) / 1000
        # ionic strength outside. /1000 is for things being in micromolar
        Ii = 0.5 * (self.nass + self.kss + cli + 4 * self.cass) / 1000
        #  The ionic strength is too high for basic DebHuc. We'll use Davies
        dielConstant = 74  # water at 37°.
        temp = 310  # body temp in kelvins.
        constA = 1.82 * 10 ** 6 * (dielConstant * temp) ** (-1.5)

        gamma_cai = exp(-constA * 4 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
        gamma_cao = exp(-constA * 4 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))
        gamma_nai = exp(-constA * 1 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
        gamma_nao = exp(-constA * 1 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))
        gamma_ki = exp(-constA * 1 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
        gamma_kao = exp(-constA * 1 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))

        PhiCaL_ss = 4.0 * self.vffrt * (gamma_cai * self.cass * exp(2.0 * self.vfrt) - gamma_cao * self.cao) / (
                exp(2.0 * self.vfrt) - 1.0)
        PhiCaNa_ss = 1.0 * self.vffrt * (gamma_nai * self.nass * exp(1.0 * self.vfrt) - gamma_nao * self.nao) / (
                exp(1.0 * self.vfrt) - 1.0)
        PhiCaK_ss = 1.0 * self.vffrt * (gamma_ki * self.kss * exp(1.0 * self.vfrt) - gamma_kao * self.ko) / (
                exp(1.0 * self.vfrt) - 1.0)

        # #  Myo driving force
        # ionic strength outside. /1000 is for things being in micromolar
        Io = 0.5 * (self.nao + self.ko + clo + 4 * self.cao) / 1000
        # ionic strength inside. /1000 is for things being in micromolar
        Ii = 0.5 * (self.nai + self.ki + cli + 4 * self.cai) / 1000

        gamma_cai = exp(-constA * 4 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
        gamma_cao = exp(-constA * 4 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))
        gamma_nai = exp(-constA * 1 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
        gamma_nao = exp(-constA * 1 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))
        gamma_ki = exp(-constA * 1 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
        gamma_kao = exp(-constA * 1 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))

        gammaCaoMyo = gamma_cao
        gammaCaiMyo = gamma_cai

        PhiCaL_i = 4.0 * self.vffrt * (gamma_cai * self.cai * exp(2.0 * self.vfrt) - gamma_cao * self.cao) / (
                exp(2.0 * self.vfrt) - 1.0)
        PhiCaNa_i = 1.0 * self.vffrt * (gamma_nai * self.nai * exp(1.0 * self.vfrt) - gamma_nao * self.nao) / (
                exp(1.0 * self.vfrt) - 1.0)
        PhiCaK_i = 1.0 * self.vffrt * (gamma_ki * self.ki * exp(1.0 * self.vfrt) - gamma_kao * self.ko) / (
                exp(1.0 * self.vfrt) - 1.0)
        # #  The rest
        # This is using one of our potentially inferred parameters
        PCa = self.P_Cab * self.multipliers["I_CaL"]

        if self.celltype == 1:
            PCa = PCa * 1.2
        elif self.celltype == 2:
            PCa = PCa * 2

        PCap = 1.1 * PCa
        PCaNa = 0.00125 * PCa
        PCaK = 3.574e-4 * PCa
        PCaNap = 0.00125 * PCap
        PCaKp = 3.574e-4 * PCap

        ICaL_ss = (1.0 - self.fICaLp) * PCa * PhiCaL_ss * self.d * (
                f * (1.0 - self.nca) + self.jca * fca * self.nca) + self.fICaLp * PCap * PhiCaL_ss * self.d * (
                          fp * (1.0 - self.nca) + self.jca * fcap * self.nca)
        ICaNa_ss = (1.0 - self.fICaLp) * PCaNa * PhiCaNa_ss * self.d * (
                f * (1.0 - self.nca) + self.jca * fca * self.nca) + self.fICaLp * PCaNap * PhiCaNa_ss * self.d * (
                           fp * (1.0 - self.nca) + self.jca * fcap * self.nca)
        ICaK_ss = (1.0 - self.fICaLp) * PCaK * PhiCaK_ss * self.d * (
                f * (1.0 - self.nca) + self.jca * fca * self.nca) + self.fICaLp * PCaKp * PhiCaK_ss * self.d * (
                          fp * (1.0 - self.nca) + self.jca * fcap * self.nca)

        ICaL_i = (1.0 - self.fICaLp) * PCa * PhiCaL_i * self.d * (
                f * (1.0 - self.nca_i) + self.jca * fca * self.nca_i) + self.fICaLp * PCap * PhiCaL_i * self.d * (
                         fp * (1.0 - self.nca_i) + self.jca * fcap * self.nca_i)
        ICaNa_i = (1.0 - self.fICaLp) * PCaNa * PhiCaNa_i * self.d * (f * (
                1.0 - self.nca_i) + self.jca * fca * self.nca_i) + self.fICaLp * PCaNap * PhiCaNa_i * self.d * (
                          fp * (1.0 - self.nca_i) + self.jca * fcap * self.nca_i)
        ICaK_i = (1.0 - self.fICaLp) * PCaK * PhiCaK_i * self.d * (
                f * (1.0 - self.nca_i) + self.jca * fca * self.nca_i) + self.fICaLp * PCaKp * PhiCaK_i * self.d * (
                         fp * (1.0 - self.nca_i) + self.jca * fcap * self.nca_i)

        ICaL_fractionSS = self.concs_and_fractions["ICaL_fractionSS"]
        #  And we weight ICaL (in ss) and ICaL_i
        ICaL_i = ICaL_i * (1 - ICaL_fractionSS)
        ICaNa_i = ICaNa_i * (1 - ICaL_fractionSS)
        ICaK_i = ICaK_i * (1 - ICaL_fractionSS)
        ICaL_ss = ICaL_ss * ICaL_fractionSS
        ICaNa_ss = ICaNa_ss * ICaL_fractionSS
        ICaK_ss = ICaK_ss * ICaL_fractionSS

        return (ICaL_ss, ICaNa_ss, ICaK_ss, ICaL_i, ICaNa_i, ICaK_i, dd, dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i,
                dffp, dfcafp, PhiCaL_ss, PhiCaL_i, gammaCaoMyo, gammaCaiMyo)

    def getIKr_ORd2011_MM(self):
        """Variant based on Lu-Vandenberg"""

        # no channels blocked in via the mechanism of specific MM states
        vfrt = self.v * self.F / (self.R * self.T)

        #  transition rates
        #  from c0 to c1 in l-v model,
        alpha = 0.1161 * exp(0.2990 * vfrt)
        #  from c1 to c0 in l-v/
        beta = 0.2442 * exp(-1.604 * vfrt)

        #  from c1 to c2 in l-v/
        alpha1 = 1.25 * 0.1235
        #  from c2 to c1 in l-v/
        beta1 = 0.1911

        #  from c2 to o/           c1 to o
        alpha2 = 0.0578 * exp(0.9710 * vfrt)  #
        #  from o to c2/
        beta2 = 0.349e-3 * exp(-1.062 * vfrt)  #

        #  from o to i
        alphai = 0.2533 * exp(0.5953 * vfrt)  #
        #  from i to o
        betai = 1.25 * 0.0522 * exp(-0.8209 * vfrt)

        #  from c2 to i (from c1 in orig)
        alphac2ToI = 0.52e-4 * exp(1.525 * vfrt)
        betaItoC2 = (beta2 * betai * alphac2ToI) / (alpha2 * alphai)

        dc0 = self.ikr_c1 * beta - self.ikr_c0 * alpha  # delta for c0
        dc1 = self.ikr_c0 * alpha + self.ikr_c2 * beta1 - self.ikr_c1 * (beta + alpha1)  # c1
        dc2 = self.ikr_c1 * alpha1 + self.ikr_o * beta2 + self.ikr_i * betaItoC2 - self.ikr_c2 * (
                beta1 + alpha2 + alphac2ToI)  # subtraction is into c2, to o, to i. #  c2
        do = self.ikr_c2 * alpha2 + self.ikr_i * betai - self.ikr_o * (beta2 + alphai)
        di = self.ikr_c2 * alphac2ToI + self.ikr_o * alphai - self.ikr_i * (betaItoC2 + betai)

        # Below is using one of our potentially inferred parameters
        GKr = self.G_Kr * sqrt(self.ko / 5) * self.multipliers[
            "I_Kr"]  # 1st element compensates for change to ko (sqrt(5/5.4)* 0.0362)
        if self.celltype == 1:
            GKr = GKr * 1.3
        elif self.celltype == 2:
            GKr = GKr * 0.8

        IKr = GKr * self.ikr_o * (self.v - self.EK)
        return IKr, dc0, dc1, dc2, do, di

    def getIKs_ORd2011(self):
        """I_Ks from ORd model"""

        # calculate IKs
        xs1ss = 1.0 / (1.0 + exp((-(self.v + 11.60)) / 8.932))
        txs1 = 817.3 + 1.0 / (2.326e-4 * exp((self.v + 48.28) / 17.80) + 0.001292 * exp((-(self.v + 210.0)) / 230.0))
        dxs1 = (xs1ss - self.xs1) / txs1
        xs2ss = xs1ss
        txs2 = 1.0 / (0.01 * exp((self.v - 50.0) / 20.0) + 0.0193 * exp((-(self.v + 66.54)) / 31.0))
        dxs2 = (xs2ss - self.xs2) / txs2
        KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / self.cai) ** 1.4)
        GKs = 0.0011 * self.multipliers["I_Ks"]
        if self.celltype == 1:
            GKs = GKs * 1.4
        IKs = GKs * KsCa * self.xs1 * self.xs2 * (self.v - self.EKs)

        return IKs, dxs1, dxs2

    def getIK1_CRLP(self):
        """IK1"""

        aK1 = 4.094 / (1 + exp(0.1217 * (self.v - self.EK - 49.934)))
        bK1 = (15.72 * exp(0.0674 * (self.v - self.EK - 3.257)) + exp(0.0618 * (self.v - self.EK - 594.31))) / (
                1 + exp(-0.1629 * (self.v - self.EK + 14.207)))
        K1ss = aK1 / (aK1 + bK1)

        GK1 = self.multipliers["I_K1"] * 0.6992  # 0.7266 # * sqrt(5/5.4))
        if self.celltype == 1:
            GK1 = GK1 * 1.2
        elif self.celltype == 2:
            GK1 = GK1 * 1.3
        IK1 = GK1 * sqrt(self.ko / 5) * K1ss * (self.v - self.EK)

        return IK1

    def getINaCa_ORd2011(self):
        """Sodium calcium exchanger"""

        zca = 2.0
        kna1 = 15.0
        kna2 = 5.0
        kna3 = 88.12
        kasymm = 12.5
        wna = 6.0e4
        wca = 6.0e4
        wnaca = 5.0e3
        kcaon = 1.5e6
        kcaoff = 5.0e3
        qna = 0.5224
        qca = 0.1670
        hca = exp((qca * self.v * self.F) / (self.R * self.T))
        hna = exp((qna * self.v * self.F) / (self.R * self.T))
        h1 = 1 + self.nai / kna3 * (1 + hna)
        h2 = (self.nai * hna) / (kna3 * h1)
        h3 = 1.0 / h1
        h4 = 1.0 + self.nai / kna1 * (1 + self.nai / kna2)
        h5 = self.nai * self.nai / (h4 * kna1 * kna2)
        h6 = 1.0 / h4
        h7 = 1.0 + self.nao / kna3 * (1.0 + 1.0 / hna)
        h8 = self.nao / (kna3 * hna * h7)
        h9 = 1.0 / h7
        h10 = kasymm + 1.0 + self.nao / kna1 * (1.0 + self.nao / kna2)
        h11 = self.nao * self.nao / (h10 * kna1 * kna2)
        h12 = 1.0 / h10
        k1 = h12 * self.cao * kcaon
        k2 = kcaoff
        k3p = h9 * wca
        k3pp = h8 * wnaca
        k3 = k3p + k3pp
        k4p = h3 * wca / hca
        k4pp = h2 * wnaca
        k4 = k4p + k4pp
        k5 = kcaoff
        k6 = h6 * self.cai * kcaon
        k7 = h5 * h2 * wna
        k8 = h8 * h11 * wna
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        KmCaAct = 150.0e-6
        allo = 1.0 / (1.0 + (KmCaAct / self.cai) ** 2.0)
        zna = 1.0
        JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
        JncxCa = E2 * k2 - E1 * k1
        Gncx = 0.0034 * self.multipliers["I_NaCa"]
        if self.celltype == 1:
            Gncx = Gncx * 1.1
        elif self.celltype == 2:
            Gncx = Gncx * 1.4
        INaCa_i = (1 - self.concs_and_fractions["INaCa_fractionSS"]) * Gncx * allo * (zna * JncxNa + zca * JncxCa)

        # calculate INaCa_ss
        h1 = 1 + self.nass / kna3 * (1 + hna)
        h2 = (self.nass * hna) / (kna3 * h1)
        h3 = 1.0 / h1
        h4 = 1.0 + self.nass / kna1 * (1 + self.nass / kna2)
        h5 = self.nass * self.nass / (h4 * kna1 * kna2)
        h6 = 1.0 / h4
        h7 = 1.0 + self.nao / kna3 * (1.0 + 1.0 / hna)
        h8 = self.nao / (kna3 * hna * h7)
        h9 = 1.0 / h7
        h10 = kasymm + 1.0 + self.nao / kna1 * (1 + self.nao / kna2)
        h11 = self.nao * self.nao / (h10 * kna1 * kna2)
        h12 = 1.0 / h10
        k1 = h12 * self.cao * kcaon
        k2 = kcaoff
        k3p = h9 * wca
        k3pp = h8 * wnaca
        k3 = k3p + k3pp
        k4p = h3 * wca / hca
        k4pp = h2 * wnaca
        k4 = k4p + k4pp
        k5 = kcaoff
        k6 = h6 * self.cass * kcaon
        k7 = h5 * h2 * wna
        k8 = h8 * h11 * wna
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        KmCaAct = 150.0e-6
        allo = 1.0 / (1.0 + (KmCaAct / self.cass) ** 2.0)
        JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
        JncxCa = E2 * k2 - E1 * k1
        INaCa_ss = self.concs_and_fractions["INaCa_fractionSS"] * Gncx * allo * (zna * JncxNa + zca * JncxCa)

        return INaCa_i, INaCa_ss

    def getINaK_ORd2011(self):
        """I_NaK"""
        zna = 1.0
        k1p = 949.5
        k1m = 182.4
        k2p = 687.2
        k2m = 39.4
        k3p = 1899.0
        k3m = 79300.0
        k4p = 639.0
        k4m = 40.0
        Knai0 = 9.073
        Knao0 = 27.78
        delta = -0.1550
        Knai = Knai0 * exp((delta * self.v * self.F) / (3.0 * self.R * self.T))
        Knao = Knao0 * exp(((1.0 - delta) * self.v * self.F) / (3.0 * self.R * self.T))
        Kki = 0.5
        Kko = 0.3582
        MgADP = 0.05
        MgATP = 9.8
        Kmgatp = 1.698e-7
        H = 1.0e-7
        eP = 4.2
        Khp = 1.698e-7
        Knap = 224.0
        Kxkur = 292.0
        P = eP / (1.0 + H / Khp + self.nai / Knap + self.ki / Kxkur)
        a1 = (k1p * (self.nai / Knai) ** 3.0) / ((1.0 + self.nai / Knai) ** 3.0 + (1.0 + self.ki / Kki) ** 2.0 - 1.0)
        b1 = k1m * MgADP
        a2 = k2p
        b2 = (k2m * (self.nao / Knao) ** 3.0) / ((1.0 + self.nao / Knao) ** 3.0 + (1.0 + self.ko / Kko) ** 2.0 - 1.0)
        a3 = (k3p * (self.ko / Kko) ** 2.0) / ((1.0 + self.nao / Knao) ** 3.0 + (1.0 + self.ko / Kko) ** 2.0 - 1.0)
        b3 = (k3m * P * H) / (1.0 + MgATP / Kmgatp)
        a4 = (k4p * MgATP / Kmgatp) / (1.0 + MgATP / Kmgatp)
        b4 = (k4m * (self.ki / Kki) ** 2.0) / ((1.0 + self.nai / Knai) ** 3.0 + (1.0 + self.ki / Kki) ** 2.0 - 1.0)
        x1 = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2
        x2 = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4
        x3 = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1
        x4 = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        zk = 1.0
        JnakNa = 3.0 * (E1 * a3 - E2 * b3)
        JnakK = 2.0 * (E4 * b1 - E3 * a1)
        Pnak = 15.4509 * self.multipliers["I_NaK"]
        if self.celltype == 1:
            Pnak = Pnak * 0.9
        elif self.celltype == 2:
            Pnak = Pnak * 0.7
        INaK = Pnak * (zna * JnakNa + zk * JnakK)

        return INaK

    def getJrel_ORd2011(self, ICaL, fJrelp):
        """J_rel"""
        jsrMidpoint = 1.7

        bt = 4.75
        a_rel = 0.5 * bt
        Jrel_inf = a_rel * (-ICaL) / (1.0 + (jsrMidpoint / self.cajsr) ** 8.0)
        if self.celltype == 2:
            Jrel_inf = Jrel_inf * 1.7
        tau_rel = bt / (1.0 + 0.0123 / self.cajsr)

        if tau_rel < 0.001:
            tau_rel = 0.001

        dJrelnp = (Jrel_inf - self.Jrel_np) / tau_rel
        btp = 1.25 * bt
        a_relp = 0.5 * btp
        Jrel_infp = a_relp * (-ICaL) / (1.0 + (jsrMidpoint / self.cajsr) ** 8.0)
        if self.celltype == 2:
            Jrel_infp = Jrel_infp * 1.7
        tau_relp = btp / (1.0 + 0.0123 / self.cajsr)

        if tau_relp < 0.001:
            tau_relp = 0.001

        dJrelp = (Jrel_infp - self.Jrel_p) / tau_relp
        Jrel = self.multipliers["J_rel"] * 1.5378 * ((1.0 - fJrelp) * self.Jrel_np + fJrelp * self.Jrel_p)

        return Jrel, dJrelnp, dJrelp

    def getJup_ORd2011(self, fJupp):
        """J_up and J_leak"""

        # calculate serca pump, ca uptake flux
        Jupnp = self.multipliers["J_up"] * 0.005425 * self.cai / (self.cai + 0.00092)
        Jupp = self.multipliers["J_up"] * 2.75 * 0.005425 * self.cai / (self.cai + 0.00092 - 0.00017)
        if self.celltype == 1:
            Jupnp = Jupnp * 1.3
            Jupp = Jupp * 1.3

        Jleak = self.multipliers["J_up"] * 0.0048825 * self.cansr / 15.0
        Jup = (1.0 - fJupp) * Jupnp + fJupp * Jupp - Jleak

        return Jup, Jleak
