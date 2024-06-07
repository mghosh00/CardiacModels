"""
This file contains the ToR-ORd model, to be used in inference.
"""
from numpy import exp, log, sqrt
from numba import jit


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


@jit
def evaluate(t, y, G_Na, P_Cab, G_Kr, G_ClCa, G_Clb, G_to, stim_duration,
             stim_amplitude, multipliers, concs_and_fractions, celltype=0,
             flag_ode=True):
    # Initial parameters
    # t = t
    # G_Na = G_Na
    # P_Cab = P_Cab
    # G_Kr = G_Kr
    # G_ClCa = G_ClCa
    # G_Clb = G_Clb
    # stim_duration = stim_duration
    # stim_amplitude = stim_amplitude
    # multipliers = multipliers
    # concs_and_fractions = concs_and_fractions
    nao = concs_and_fractions[0]
    cao = concs_and_fractions[1]
    ko = concs_and_fractions[2]
    # celltype = celltype
    # flag_ode = flag_ode

    # physical constants
    R = 8314.0
    T = 310.0
    F = 96485.0

    # cell geometry
    L = 0.01
    rad = 0.0011
    vcell = 1000 * 3.14 * rad * rad * L
    Ageo = 2 * 3.14 * rad * rad + 2 * 3.14 * rad * L
    Acap = 2 * Ageo
    vmyo = 0.68 * vcell
    vnsr = 0.0552 * vcell
    vjsr = 0.0048 * vcell
    vss = 0.02 * vcell

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # membrane potential
    v = y[0]

    # ionic compartment concentrations
    nai = y[1]
    nass = y[2]
    ki = y[3]
    kss = y[4]
    cai = y[5]
    cass = y[6]
    cansr = y[7]
    cajsr = y[8]

    # I_Na gating variables
    m = y[9]
    hp = y[10]
    h = y[11]
    j = y[12]
    jp = y[13]

    # I_NaL gating variables
    mL = y[14]
    hL = y[15]
    hLp = y[16]

    # I_to gating variables
    a = y[17]
    iF = y[18]
    iS = y[19]
    ap = y[20]
    iFp = y[21]
    iSp = y[22]

    # I_CaL gating variables
    d = y[23]
    ff = y[24]
    fs = y[25]
    fcaf = y[26]
    fcas = y[27]
    jca = y[28]
    nca = y[29]
    nca_i = y[30]
    ffp = y[31]
    fcafp = y[32]

    # I_Ks gating variables
    xs1 = y[33]
    xs2 = y[34]

    # J_rel, CaMKII and I_Kr variables
    Jrel_np = y[35]
    CaMKt = y[36]
    ikr_c0 = y[37]
    ikr_c1 = y[38]
    ikr_c2 = y[39]
    ikr_o = y[40]
    ikr_i = y[41]
    Jrel_p = y[42]

    cli = 24  # Intracellular Cl  [mM]
    clo = 150  # Extracellular Cl  [mM]
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # CaMK constants
    KmCaMK = 0.15

    aCaMK = 0.05
    bCaMK = 0.00068
    CaMKo = 0.05
    KmCaM = 0.0015
    # update CaMK
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = CaMKb + CaMKt
    dCaMKt = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # reversal potentials
    ENa = (R * T / F) * log(nao / nai)
    EK = (R * T / F) * log(ko / ki)
    PKNa = 0.01833
    EKs = (R * T / F) * log((ko + PKNa * nao) / (ki + PKNa * nai))

    # convenient shorthand calculations
    vffrt = v * F * F / (R * T)
    vfrt = v * F / (R * T)
    frt = F / (R * T)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    fINap = (1.0 / (1.0 + KmCaMK / CaMKa))
    fINaLp = (1.0 / (1.0 + KmCaMK / CaMKa))
    fItop = (1.0 / (1.0 + KmCaMK / CaMKa))
    fICaLp = (1.0 / (1.0 + KmCaMK / CaMKa))

    # #  INa
    INa, dm, dh, dhp, dj, djp = getINa_Grandi(v, m, h, hp, j, jp, fINap, ENa, multipliers[1], G_Na)

    # #  INaL
    INaL, dmL, dhL, dhLp = getINaL_ORd2011(v, mL, hL, hLp, fINaLp, ENa, celltype, multipliers[3])

    # #  ITo
    Ito, da, diF, diS, dap, diFp, diSp = getITo_ORd2011(v, a, iF, iS, ap, iFp, iSp, fItop, EK, celltype,
                                                        multipliers[2], G_to)

    # #  ICaL
    (ICaL_ss, ICaNa_ss, ICaK_ss, ICaL_i, ICaNa_i, ICaK_i, dd, dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i, dffp,
     dfcafp, PhiCaL_ss, PhiCaL_i, gammaCaoMyo, gammaCaiMyo) = getICaL_ORd2011_jt(v, d, ff, fs, fcaf, fcas, jca, nca,
                                                                                 nca_i,
                                                                                 ffp, fcafp, fICaLp, cai, cass, cao,
                                                                                 nai,
                                                                                 nass, nao, ki, kss, ko, cli, clo,
                                                                                 celltype,
                                                                                 concs_and_fractions[3],
                                                                                 multipliers[0], P_Cab)

    ICaL = ICaL_ss + ICaL_i
    ICaNa = ICaNa_ss + ICaNa_i
    ICaK = ICaK_ss + ICaK_i
    ICaL_tot = ICaL + ICaNa + ICaK
    # #  IKr
    IKr, dt_ikr_c0, dt_ikr_c1, dt_ikr_c2, dt_ikr_o, dt_ikr_i = getIKr_ORd2011_MM(v, ikr_c0, ikr_c1, ikr_c2, ikr_o,
                                                                                 ikr_i, ko, EK, celltype,
                                                                                 multipliers[4], G_Kr)

    # #  IKs
    IKs, dxs1, dxs2 = getIKs_ORd2011(v, xs1, xs2, cai, EKs, celltype, multipliers[5])

    # #  IK1
    IK1 = getIK1_CRLP(v, ko, EK, celltype, multipliers[6])

    # #  INaCa
    INaCa_i, INaCa_ss = getINaCa_ORd2011(v, F, R, T, nass, nai, nao, cass, cai, cao, celltype,
                                         multipliers[8], concs_and_fractions[4])

    # #  INaK
    INaK = getINaK_ORd2011(v, F, R, T, nai, nao, ki, ko, celltype, multipliers[9])

    # #  Minor/background currents
    # calculate IKb
    xkb = 1.0 / (1.0 + exp(-(v - 10.8968) / 23.9871))
    GKb = 0.0189 * multipliers[7]
    if celltype == 1:
        GKb = GKb * 0.6
    IKb = GKb * xkb * (v - EK)

    # calculate INab
    PNab = 1.9239e-09 * multipliers[10]
    INab = PNab * vffrt * (nai * exp(vfrt) - nao) / (exp(vfrt) - 1.0)

    # calculate ICab
    PCab = 5.9194e-08 * multipliers[11]
    #
    ICab = PCab * 4.0 * vffrt * (gammaCaiMyo * cai * exp(2.0 * vfrt) - gammaCaoMyo * cao) / (
            exp(2.0 * vfrt) - 1.0)

    # calculate IpCa
    GpCa = 5e-04 * multipliers[12]
    IpCa = GpCa * cai / (0.0005 + cai)

    # #  Chloride
    #  I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current

    ecl = (R * T / F) * log(cli / clo)  # [mV]

    Fjunc = 1
    Fsl = 1 - Fjunc  # fraction in SS and in myoplasm - as per literature, I(Ca)Cl is in junctional subspace

    # Both of the below conductances are some of the inferred parameters
    GClCa = multipliers[13] * G_ClCa  # [mS/uF]
    GClB = multipliers[14] * G_Clb  # [mS/uF] #
    KdClCa = 0.1  # [mM]

    I_ClCa_junc = Fjunc * GClCa / (1 + KdClCa / cass) * (v - ecl)
    I_ClCa_sl = Fsl * GClCa / (1 + KdClCa / cai) * (v - ecl)

    I_ClCa = I_ClCa_junc + I_ClCa_sl
    I_Clbk = GClB * (v - ecl)

    # #  Calcium handling
    # calculate ryanodione receptor calcium induced calcium release from the jsr

    # #  Jrel
    fJrelp = (1.0 / (1.0 + KmCaMK / CaMKa))
    Jrel, dJrel_np, dJrel_p = getJrel_ORd2011(Jrel_np, Jrel_p, ICaL_ss, cass, cajsr, fJrelp, celltype,
                                              multipliers[15])

    fJupp = (1.0 / (1.0 + KmCaMK / CaMKa))
    Jup, Jleak = getJup_ORd2011(cai, cansr, fJupp, celltype, multipliers[16])

    # calculate translocation flux
    Jtr = (cansr - cajsr) / 60

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # #  calculate the stimulus current, Istim
    amp = stim_amplitude
    duration = stim_duration
    if t <= duration:
        Istim = amp
    else:
        Istim = 0.0

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # update the membrane voltage

    dv = -(INa + INaL + Ito + ICaL + ICaNa + ICaK + IKr + IKs + IK1 + INaCa_i + INaCa_ss + INaK + INab + IKb +
           IpCa + ICab + I_ClCa + I_Clbk + Istim)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # calculate diffusion fluxes
    JdiffNa = (nass - nai) / 2.0
    JdiffK = (kss - ki) / 2.0
    Jdiff = (cass - cai) / 0.2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # calcium buffer constants
    cmdnmax = 0.05
    if celltype == 1:
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
    dnai = -(ICaNa_i + INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + INab) * Acap / (
            F * vmyo) + JdiffNa * vss / vmyo
    dnass = -(ICaNa_ss + 3.0 * INaCa_ss) * Acap / (F * vss) - JdiffNa

    dki = -(ICaK_i + Ito + IKr + IKs + IK1 + IKb + Istim - 2.0 * INaK) * Acap / (
            F * vmyo) + JdiffK * vss / vmyo
    dkss = -ICaK_ss * Acap / (F * vss) - JdiffK

    Bcai = (1.0 / (1.0 + cmdnmax * kmcmdn / (kmcmdn + cai) ** 2.0
                   + trpnmax * kmtrpn / (kmtrpn + cai) ** 2.0))
    dcai = Bcai * (-(ICaL_i + IpCa + ICab - 2.0 * INaCa_i) * Acap / (
            2.0 * F * vmyo) - Jup * vnsr / vmyo + Jdiff * vss / vmyo)

    Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + cass) ** 2.0 + BSLmax * KmBSL / (KmBSL + cass) ** 2.0)
    dcass = Bcass * (-(ICaL_ss - 2.0 * INaCa_ss) * Acap / (
            2.0 * F * vss) + Jrel * vjsr / vss - Jdiff)

    dcansr = Jup - Jtr * vjsr / vnsr

    Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + cajsr) ** 2.0)
    dcajsr = Bcajsr * (Jtr - Jrel)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # output the state vector when ode_flag==1, and the calculated currents and fluxes otherwise
    if flag_ode == 1:
        return [dv, dnai, dnass, dki, dkss, dcai, dcass, dcansr, dcajsr, dm, dhp,
                dh, dj, djp, dmL, dhL, dhLp, da, diF, diS, dap, diFp, diSp, dd,
                dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i, dffp, dfcafp, dxs1,
                dxs2, dJrel_np, dCaMKt, dt_ikr_c0, dt_ikr_c1, dt_ikr_c2, dt_ikr_o,
                dt_ikr_i, dJrel_p]
    else:
        return [INa, INaL, Ito, ICaL, IKr, IKs, IK1, INaCa_i, INaCa_ss, INaK,
                IKb, INab, ICab, IpCa, Jdiff, JdiffNa, JdiffK, Jup, Jleak, Jtr,
                Jrel, CaMKa, Istim, fINap, fINaLp, fICaLp,
                fJrelp, fJupp, cajsr, cansr, PhiCaL_ss, v,
                ICaL_i, I_ClCa, I_Clbk, ICaL_tot]


@jit
def getINa_Grandi(v, m, h, hp, j, jp, fINap, ENa, INa_Multiplier, G_Na):
    """INa formulations"""

    #  The Grandi implementation updated with INa phosphorylation.
    # #  m gate
    mss = 1 / ((1 + exp(-(56.86 + v) / 9.03)) ** 2)
    taum = 0.1292 * exp(-((v + 45.79) / 15.54) ** 2) + 0.06487 * exp(-((v - 4.823) / 51.12) ** 2)
    dm = (mss - m) / taum

    # #  h gate
    ah = (v >= -40) * 0 + (v < -40) * (0.057 * exp(-(v + 80) / 6.8))
    bh = ((v >= -40) * (0.77 / (0.13 * (1 + exp(-(v + 10.66) / 11.1))))
          + (v < -40) * (2.7 * exp(0.079 * v) + 3.1 * 10 ** 5 * exp(0.3485 * v)))
    tauh = 1 / (ah + bh)
    hss = 1 / ((1 + exp((v + 71.55) / 7.43)) ** 2)
    dh = (hss - h) / tauh
    # #  j gate
    aj = ((v >= -40) * 0
          + (v < -40) * (((-2.5428 * 10 ** 4 * exp(0.2444 * v)
                           - 6.948 * 10 ** -6 * exp(-0.04391 * v))
                          * (v + 37.78)) / (1 + exp(0.311 * (v + 79.23)))))
    bj = ((v >= -40) * ((0.6 * exp(0.057 * v)) / (1 + exp(-0.1 * (v + 32))))
          + (v < -40) * ((0.02424 * exp(-0.01052 * v)) / (1 + exp(-0.1378 * (v + 40.14)))))
    tauj = 1 / (aj + bj)
    jss = 1 / ((1 + exp((v + 71.55) / 7.43)) ** 2)
    dj = (jss - j) / tauj

    # #  h phosphorylated
    hssp = 1 / ((1 + exp((v + 71.55 + 6) / 7.43)) ** 2)
    dhp = (hssp - hp) / tauh
    # #  j phosphorylated
    taujp = 1.46 * tauj
    djp = (jss - jp) / taujp

    # Below is using one of our potentially inferred parameters, G_Na
    INa = INa_Multiplier * G_Na * (v - ENa) * m ** 3.0 * (
            (1.0 - fINap) * h * j + fINap * hp * jp)

    return INa, dm, dh, dhp, dj, djp


@jit
def getINaL_ORd2011(v, mL, hL, hLp, fINaLp, ENa, celltype, INaL_Multiplier):
    """INaL formulations"""
    # calculate INaL
    mLss = 1.0 / (1.0 + exp((-(v + 42.85)) / 5.264))
    tm = 0.1292 * exp(-((v + 45.79) / 15.54) ** 2) + 0.06487 * exp(-((v - 4.823) / 51.12) ** 2)
    tmL = tm
    dmL = (mLss - mL) / tmL
    hLss = 1.0 / (1.0 + exp((v + 87.61) / 7.488))
    thL = 200.0
    dhL = (hLss - hL) / thL
    hLssp = 1.0 / (1.0 + exp((v + 93.81) / 7.488))
    thLp = 3.0 * thL
    dhLp = (hLssp - hLp) / thLp
    GNaL = 0.0279 * INaL_Multiplier
    if celltype == 1:
        GNaL = GNaL * 0.6

    INaL = GNaL * (v - ENa) * mL * ((1.0 - fINaLp) * hL + fINaLp * hLp)
    return INaL, dmL, dhL, dhLp


@jit
def getITo_ORd2011(v, a, iF, iS, ap, iFp, iSp, fItop, EK, celltype, Ito_Multiplier, G_to):
    """Ito formulations"""
    # calculate Ito
    ass = 1.0 / (1.0 + exp((-(v - 14.34)) / 14.82))
    ta = 1.0515 / (1.0 / (1.2089 * (1.0 + exp(-(v - 18.4099) / 29.3814))) + 3.5 / (
            1.0 + exp((v + 100.0) / 29.3814)))
    da = (ass - a) / ta
    iss = 1.0 / (1.0 + exp((v + 43.94) / 5.711))
    if celltype == 1:
        delta_epi = 1.0 - (0.95 / (1.0 + exp((v + 70.0) / 5.0)))
    else:
        delta_epi = 1.0
    tiF = 4.562 + 1 / (0.3933 * exp((-(v + 100.0)) / 100.0) + 0.08004 * exp((v + 50.0) / 16.59))
    tiS = 23.62 + 1 / (0.001416 * exp((-(v + 96.52)) / 59.05) + 1.780e-8 * exp((v + 114.1) / 8.079))
    tiF = tiF * delta_epi
    tiS = tiS * delta_epi
    AiF = 1.0 / (1.0 + exp((v - 213.6) / 151.2))
    AiS = 1.0 - AiF
    diF = (iss - iF) / tiF
    diS = (iss - iS) / tiS
    i = AiF * iF + AiS * iS
    assp = 1.0 / (1.0 + exp((-(v - 24.34)) / 14.82))
    dap = (assp - ap) / ta
    dti_develop = 1.354 + 1.0e-4 / (exp((v - 167.4) / 15.89) + exp(-(v - 12.23) / 0.2154))
    dti_recover = 1.0 - 0.5 / (1.0 + exp((v + 70.0) / 20.0))
    tiFp = dti_develop * dti_recover * tiF
    tiSp = dti_develop * dti_recover * tiS
    diFp = (iss - iFp) / tiFp
    diSp = (iss - iSp) / tiSp
    ip = AiF * iFp + AiS * iSp

    # One of the potentially inferred constants
    Gto = G_to * Ito_Multiplier
    if celltype == 1:
        Gto = Gto * 2.0
    elif celltype == 2:
        Gto = Gto * 2.0

    Ito = Gto * (v - EK) * ((1.0 - fItop) * a * i + fItop * ap * ip)

    return Ito, da, diF, diS, dap, diFp, diSp


#  a variant updated by jakub, using a changed activation curve
#  it computes both ICaL in subspace and myoplasm (_i)
@jit
def getICaL_ORd2011_jt(v, d, ff, fs, fcaf, fcas, jca, nca, nca_i, ffp, fcafp,
                       fICaLp, cai, cass, cao, nai, nass, nao, ki, kss, ko, cli, clo, celltype, ICaL_fractionSS,
                       ICaL_PCaMultiplier, P_Cab):
    """a variant updated by jakub, using a changed activation curve -
    it computes both ICaL in subspace and myoplasm (_i)"""

    # physical constants
    R = 8314.0
    T = 310.0
    F = 96485.0
    vffrt = v * F * F / (R * T)
    vfrt = v * F / (R * T)

    # calculate ICaL, ICaNa, ICaK

    dss = 1.0763 * exp(-1.0070 * exp(-0.0829 * v))  # magyar
    if v > 31.4978:  # activation cannot be greater than 1
        dss = 1

    td = 0.6 + 1.0 / (exp(-0.05 * (v + 6.0)) + exp(0.09 * (v + 14.0)))

    dd = (dss - d) / td
    fss = 1.0 / (1.0 + exp((v + 19.58) / 3.696))
    tff = 7.0 + 1.0 / (0.0045 * exp(-(v + 20.0) / 10.0) + 0.0045 * exp((v + 20.0) / 10.0))
    tfs = 1000.0 + 1.0 / (0.000035 * exp(-(v + 5.0) / 4.0) + 0.000035 * exp((v + 5.0) / 6.0))
    Aff = 0.6
    Afs = 1.0 - Aff
    dff = (fss - ff) / tff
    dfs = (fss - fs) / tfs
    f = Aff * ff + Afs * fs
    fcass = fss
    tfcaf = 7.0 + 1.0 / (0.04 * exp(-(v - 4.0) / 7.0) + 0.04 * exp((v - 4.0) / 7.0))
    tfcas = 100.0 + 1.0 / (0.00012 * exp(-v / 3.0) + 0.00012 * exp(v / 7.0))

    Afcaf = 0.3 + 0.6 / (1.0 + exp((v - 10.0) / 10.0))

    Afcas = 1.0 - Afcaf
    dfcaf = (fcass - fcaf) / tfcaf
    dfcas = (fcass - fcas) / tfcas
    fca = Afcaf * fcaf + Afcas * fcas

    tjca = 75
    jcass = 1.0 / (1.0 + exp((v + 18.08) / 2.7916))
    djca = (jcass - jca) / tjca
    tffp = 2.5 * tff
    dffp = (fss - ffp) / tffp
    fp = Aff * ffp + Afs * fs
    tfcafp = 2.5 * tfcaf
    dfcafp = (fcass - fcafp) / tfcafp
    fcap = Afcaf * fcafp + Afcas * fcas

    # #  SS nca
    Kmn = 0.002
    k2n = 500.0
    km2n = jca * 1
    anca = 1.0 / (k2n / km2n + (1.0 + Kmn / cass) ** 4.0)
    dnca = anca * k2n - nca * km2n

    # #  myoplasmic nca
    anca_i = 1.0 / (k2n / km2n + (1.0 + Kmn / cai) ** 4.0)
    dnca_i = anca_i * k2n - nca_i * km2n

    # #  SS driving force
    clo = 150
    cli = 24
    # ionic strength outside. /1000 is for things being in micromolar
    Io = 0.5 * (nao + ko + clo + 4 * cao) / 1000
    # ionic strength outside. /1000 is for things being in micromolar
    Ii = 0.5 * (nass + kss + cli + 4 * cass) / 1000
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

    PhiCaL_ss = 4.0 * vffrt * (gamma_cai * cass * exp(2.0 * vfrt) - gamma_cao * cao) / (
            exp(2.0 * vfrt) - 1.0)
    PhiCaNa_ss = 1.0 * vffrt * (gamma_nai * nass * exp(1.0 * vfrt) - gamma_nao * nao) / (
            exp(1.0 * vfrt) - 1.0)
    PhiCaK_ss = 1.0 * vffrt * (gamma_ki * kss * exp(1.0 * vfrt) - gamma_kao * ko) / (
            exp(1.0 * vfrt) - 1.0)

    # #  Myo driving force
    # ionic strength outside. /1000 is for things being in micromolar
    Io = 0.5 * (nao + ko + clo + 4 * cao) / 1000
    # ionic strength inside. /1000 is for things being in micromolar
    Ii = 0.5 * (nai + ki + cli + 4 * cai) / 1000

    gamma_cai = exp(-constA * 4 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
    gamma_cao = exp(-constA * 4 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))
    gamma_nai = exp(-constA * 1 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
    gamma_nao = exp(-constA * 1 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))
    gamma_ki = exp(-constA * 1 * (sqrt(Ii) / (1 + sqrt(Ii)) - 0.3 * Ii))
    gamma_kao = exp(-constA * 1 * (sqrt(Io) / (1 + sqrt(Io)) - 0.3 * Io))

    gammaCaoMyo = gamma_cao
    gammaCaiMyo = gamma_cai

    PhiCaL_i = 4.0 * vffrt * (gamma_cai * cai * exp(2.0 * vfrt) - gamma_cao * cao) / (
            exp(2.0 * vfrt) - 1.0)
    PhiCaNa_i = 1.0 * vffrt * (gamma_nai * nai * exp(1.0 * vfrt) - gamma_nao * nao) / (
            exp(1.0 * vfrt) - 1.0)
    PhiCaK_i = 1.0 * vffrt * (gamma_ki * ki * exp(1.0 * vfrt) - gamma_kao * ko) / (
            exp(1.0 * vfrt) - 1.0)
    # #  The rest
    # This is using one of our potentially inferred parameters
    PCa = P_Cab * ICaL_PCaMultiplier

    if celltype == 1:
        PCa = PCa * 1.2
    elif celltype == 2:
        PCa = PCa * 2

    PCap = 1.1 * PCa
    PCaNa = 0.00125 * PCa
    PCaK = 3.574e-4 * PCa
    PCaNap = 0.00125 * PCap
    PCaKp = 3.574e-4 * PCap

    ICaL_ss = (1.0 - fICaLp) * PCa * PhiCaL_ss * d * (
            f * (1.0 - nca) + jca * fca * nca) + fICaLp * PCap * PhiCaL_ss * d * (
                      fp * (1.0 - nca) + jca * fcap * nca)
    ICaNa_ss = (1.0 - fICaLp) * PCaNa * PhiCaNa_ss * d * (
            f * (1.0 - nca) + jca * fca * nca) + fICaLp * PCaNap * PhiCaNa_ss * d * (
                       fp * (1.0 - nca) + jca * fcap * nca)
    ICaK_ss = (1.0 - fICaLp) * PCaK * PhiCaK_ss * d * (
            f * (1.0 - nca) + jca * fca * nca) + fICaLp * PCaKp * PhiCaK_ss * d * (
                      fp * (1.0 - nca) + jca * fcap * nca)

    ICaL_i = (1.0 - fICaLp) * PCa * PhiCaL_i * d * (
            f * (1.0 - nca_i) + jca * fca * nca_i) + fICaLp * PCap * PhiCaL_i * d * (
                     fp * (1.0 - nca_i) + jca * fcap * nca_i)
    ICaNa_i = (1.0 - fICaLp) * PCaNa * PhiCaNa_i * d * (f * (
            1.0 - nca_i) + jca * fca * nca_i) + fICaLp * PCaNap * PhiCaNa_i * d * (
                      fp * (1.0 - nca_i) + jca * fcap * nca_i)
    ICaK_i = (1.0 - fICaLp) * PCaK * PhiCaK_i * d * (
            f * (1.0 - nca_i) + jca * fca * nca_i) + fICaLp * PCaKp * PhiCaK_i * d * (
                     fp * (1.0 - nca_i) + jca * fcap * nca_i)

    #  And we weight ICaL (in ss) and ICaL_i
    ICaL_i = ICaL_i * (1 - ICaL_fractionSS)
    ICaNa_i = ICaNa_i * (1 - ICaL_fractionSS)
    ICaK_i = ICaK_i * (1 - ICaL_fractionSS)
    ICaL_ss = ICaL_ss * ICaL_fractionSS
    ICaNa_ss = ICaNa_ss * ICaL_fractionSS
    ICaK_ss = ICaK_ss * ICaL_fractionSS

    return (ICaL_ss, ICaNa_ss, ICaK_ss, ICaL_i, ICaNa_i, ICaK_i, dd, dff, dfs, dfcaf, dfcas, djca, dnca, dnca_i,
            dffp, dfcafp, PhiCaL_ss, PhiCaL_i, gammaCaoMyo, gammaCaiMyo)


@jit
def getIKr_ORd2011_MM(v, ikr_c0, ikr_c1, ikr_c2, ikr_o, ikr_i, ko, EK, celltype, IKr_Multiplier, G_Kr):
    """Variant based on Lu-Vandenberg"""

    # physical constants
    R = 8314.0
    T = 310.0
    F = 96485.0

    # no channels blocked in via the mechanism of specific MM states
    vfrt = v * F / (R * T)

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

    dc0 = ikr_c1 * beta - ikr_c0 * alpha  # delta for c0
    dc1 = ikr_c0 * alpha + ikr_c2 * beta1 - ikr_c1 * (beta + alpha1)  # c1
    dc2 = ikr_c1 * alpha1 + ikr_o * beta2 + ikr_i * betaItoC2 - ikr_c2 * (
            beta1 + alpha2 + alphac2ToI)  # subtraction is into c2, to o, to i. #  c2
    do = ikr_c2 * alpha2 + ikr_i * betai - ikr_o * (beta2 + alphai)
    di = ikr_c2 * alphac2ToI + ikr_o * alphai - ikr_i * (betaItoC2 + betai)

    # Below is using one of our potentially inferred parameters
    GKr = G_Kr * sqrt(ko / 5) * IKr_Multiplier  # 1st element compensates for change to ko (sqrt(5/5.4)* 0.0362)
    if celltype == 1:
        GKr = GKr * 1.3
    elif celltype == 2:
        GKr = GKr * 0.8

    IKr = GKr * ikr_o * (v - EK)
    return IKr, dc0, dc1, dc2, do, di


@jit
def getIKs_ORd2011(v, xs1, xs2, cai, EKs, celltype, IKs_Multiplier):
    """I_Ks from ORd model"""

    # calculate IKs
    xs1ss = 1.0 / (1.0 + exp((-(v + 11.60)) / 8.932))
    txs1 = 817.3 + 1.0 / (2.326e-4 * exp((v + 48.28) / 17.80) + 0.001292 * exp((-(v + 210.0)) / 230.0))
    dxs1 = (xs1ss - xs1) / txs1
    xs2ss = xs1ss
    txs2 = 1.0 / (0.01 * exp((v - 50.0) / 20.0) + 0.0193 * exp((-(v + 66.54)) / 31.0))
    dxs2 = (xs2ss - xs2) / txs2
    KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / cai) ** 1.4)
    GKs = 0.0011 * IKs_Multiplier
    if celltype == 1:
        GKs = GKs * 1.4
    IKs = GKs * KsCa * xs1 * xs2 * (v - EKs)

    return IKs, dxs1, dxs2


@jit
def getIK1_CRLP(v, ko, EK, celltype, IK1_Multiplier):
    """IK1"""

    aK1 = 4.094 / (1 + exp(0.1217 * (v - EK - 49.934)))
    bK1 = (15.72 * exp(0.0674 * (v - EK - 3.257)) + exp(0.0618 * (v - EK - 594.31))) / (
            1 + exp(-0.1629 * (v - EK + 14.207)))
    K1ss = aK1 / (aK1 + bK1)

    GK1 = IK1_Multiplier * 0.6992  # 0.7266 # * sqrt(5/5.4))
    if celltype == 1:
        GK1 = GK1 * 1.2
    elif celltype == 2:
        GK1 = GK1 * 1.3
    IK1 = GK1 * sqrt(ko / 5) * K1ss * (v - EK)

    return IK1


@jit
def getINaCa_ORd2011(v, F, R, T, nass, nai, nao, cass, cai, cao, celltype, INaCa_Multiplier, INaCa_fractionSS):
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
    hca = exp((qca * v * F) / (R * T))
    hna = exp((qna * v * F) / (R * T))
    h1 = 1 + nai / kna3 * (1 + hna)
    h2 = (nai * hna) / (kna3 * h1)
    h3 = 1.0 / h1
    h4 = 1.0 + nai / kna1 * (1 + nai / kna2)
    h5 = nai * nai / (h4 * kna1 * kna2)
    h6 = 1.0 / h4
    h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
    h8 = nao / (kna3 * hna * h7)
    h9 = 1.0 / h7
    h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
    h11 = nao * nao / (h10 * kna1 * kna2)
    h12 = 1.0 / h10
    k1 = h12 * cao * kcaon
    k2 = kcaoff
    k3p = h9 * wca
    k3pp = h8 * wnaca
    k3 = k3p + k3pp
    k4p = h3 * wca / hca
    k4pp = h2 * wnaca
    k4 = k4p + k4pp
    k5 = kcaoff
    k6 = h6 * cai * kcaon
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
    allo = 1.0 / (1.0 + (KmCaAct / cai) ** 2.0)
    zna = 1.0
    JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
    JncxCa = E2 * k2 - E1 * k1
    Gncx = 0.0034 * INaCa_Multiplier
    if celltype == 1:
        Gncx = Gncx * 1.1
    elif celltype == 2:
        Gncx = Gncx * 1.4
    INaCa_i = (1 - INaCa_fractionSS) * Gncx * allo * (zna * JncxNa + zca * JncxCa)

    # calculate INaCa_ss
    h1 = 1 + nass / kna3 * (1 + hna)
    h2 = (nass * hna) / (kna3 * h1)
    h3 = 1.0 / h1
    h4 = 1.0 + nass / kna1 * (1 + nass / kna2)
    h5 = nass * nass / (h4 * kna1 * kna2)
    h6 = 1.0 / h4
    h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
    h8 = nao / (kna3 * hna * h7)
    h9 = 1.0 / h7
    h10 = kasymm + 1.0 + nao / kna1 * (1 + nao / kna2)
    h11 = nao * nao / (h10 * kna1 * kna2)
    h12 = 1.0 / h10
    k1 = h12 * cao * kcaon
    k2 = kcaoff
    k3p = h9 * wca
    k3pp = h8 * wnaca
    k3 = k3p + k3pp
    k4p = h3 * wca / hca
    k4pp = h2 * wnaca
    k4 = k4p + k4pp
    k5 = kcaoff
    k6 = h6 * cass * kcaon
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
    allo = 1.0 / (1.0 + (KmCaAct / cass) ** 2.0)
    JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
    JncxCa = E2 * k2 - E1 * k1
    INaCa_ss = INaCa_fractionSS * Gncx * allo * (zna * JncxNa + zca * JncxCa)

    return INaCa_i, INaCa_ss


@jit
def getINaK_ORd2011(v, F, R, T, nai, nao, ki, ko, celltype, INaK_Multiplier):
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
    Knai = Knai0 * exp((delta * v * F) / (3.0 * R * T))
    Knao = Knao0 * exp(((1.0 - delta) * v * F) / (3.0 * R * T))
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
    P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)
    a1 = (k1p * (nai / Knai) ** 3.0) / ((1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0)
    b1 = k1m * MgADP
    a2 = k2p
    b2 = (k2m * (nao / Knao) ** 3.0) / ((1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0)
    a3 = (k3p * (ko / Kko) ** 2.0) / ((1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0)
    b3 = (k3m * P * H) / (1.0 + MgATP / Kmgatp)
    a4 = (k4p * MgATP / Kmgatp) / (1.0 + MgATP / Kmgatp)
    b4 = (k4m * (ki / Kki) ** 2.0) / ((1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0)
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
    Pnak = 15.4509 * INaK_Multiplier
    if celltype == 1:
        Pnak = Pnak * 0.9
    elif celltype == 2:
        Pnak = Pnak * 0.7
    INaK = Pnak * (zna * JnakNa + zk * JnakK)

    return INaK


@jit
def getJrel_ORd2011(Jrel_np, Jrel_p, ICaL, cass, cajsr, fJrelp, celltype, Jrel_Multiplier):
    """J_rel"""
    jsrMidpoint = 1.7

    bt = 4.75
    a_rel = 0.5 * bt
    Jrel_inf = a_rel * (-ICaL) / (1.0 + (jsrMidpoint / cajsr) ** 8.0)
    if celltype == 2:
        Jrel_inf = Jrel_inf * 1.7
    tau_rel = bt / (1.0 + 0.0123 / cajsr)

    if tau_rel < 0.001:
        tau_rel = 0.001

    dJrelnp = (Jrel_inf - Jrel_np) / tau_rel
    btp = 1.25 * bt
    a_relp = 0.5 * btp
    Jrel_infp = a_relp * (-ICaL) / (1.0 + (jsrMidpoint / cajsr) ** 8.0)
    if celltype == 2:
        Jrel_infp = Jrel_infp * 1.7
    tau_relp = btp / (1.0 + 0.0123 / cajsr)

    if tau_relp < 0.001:
        tau_relp = 0.001

    dJrelp = (Jrel_infp - Jrel_p) / tau_relp
    Jrel = Jrel_Multiplier * 1.5378 * ((1.0 - fJrelp) * Jrel_np + fJrelp * Jrel_p)

    return Jrel, dJrelnp, dJrelp


@jit
def getJup_ORd2011(cai, cansr, fJupp, celltype, Jup_Multiplier):
    """J_up and J_leak"""

    # calculate serca pump, ca uptake flux
    Jupnp = Jup_Multiplier * 0.005425 * cai / (cai + 0.00092)
    Jupp = Jup_Multiplier * 2.75 * 0.005425 * cai / (cai + 0.00092 - 0.00017)
    if celltype == 1:
        Jupnp = Jupnp * 1.3
        Jupp = Jupp * 1.3

    Jleak = Jup_Multiplier * 0.0048825 * cansr / 15.0
    Jup = (1.0 - fJupp) * Jupnp + fJupp * Jupp - Jleak

    return Jup, Jleak
