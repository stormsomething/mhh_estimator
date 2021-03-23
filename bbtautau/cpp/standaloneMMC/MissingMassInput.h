/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

// Class handling the input of the MissingMassCalculator
// author Michael Huebner <michael.huebner@no.spam.cern.ch>

#ifndef DITAUMASSTOOLS_MISSINGMASSINPUT_H
#define DITAUMASSTOOLS_MISSINGMASSINPUT_H

// Local include(s):
#include "HelperFunctions.h"

namespace DiTauMassTools{

class MissingMassInput {

  public:
    MissingMassInput();
    ~MissingMassInput();

    void ClearInput();
    void PrintInputInfo();

    void SetBeamEnergy(double val) { beamEnergy=val; } // beam energy
    void SetMETresSyst(int val) { METresSyst=val; } // MET resolution systematics: +/-1 sigma
    void SetMetScanParamsUE(double sumEt, double phi_scan=0.0, int data_code=0);
    void SetNjet25(int val);
    void SetSumEt(double sumEt);
    void SetVisTauVec(int i, const TLorentzVector & vec);
    void SetVisTauType(int i, int tautype);
    void SetMetVec(const TVector2 & vec);
    void SetNprong(int i, int nprong);
    void SetIsData(int val);
    void SetMHtSigma1(double val) { MHtSigma1=val; }
    void SetMHtSigma2(double val) { MHtSigma2=val; } 
    void SetMHtGaussFr(double val) { MHtGaussFr=val; }
    void SetMETScanScheme(int val) { METScanScheme=val; }
    void SetUseDefaults(int val) { fUseDefaults=val; }
    void SetUseTailCleanup(bool val) { fUseTailCleanup=val; }
    void SetUseVerbose(bool val) { fUseVerbose=val; }
    void SetLFVmode(int val) { LFVmode=val; }

    friend class MissingMassCalculatorV2;
    friend class MissingMassProb;

  private:
    TVector2 MetVec;
    TLorentzVector vistau1;
    TLorentzVector vistau2;
    int type_visTau1; // 0: l, 1:1-prong, 3:3-prong
    int type_visTau2;  // 0: l, 1:1-prong, 3:3-prong
    int Nprong_tau1;
    int Nprong_tau2;
    int dataType;
    double METcovphi;
    double METsigmaP;
    double METsigmaL;
    double SumEt;
    std::vector<TLorentzVector> jet4vecs;
    int Njet25;
    double DelPhiTT;
    double MHtSigma1; // sigma of 1st Gaussian in missing Ht resolution
    double MHtSigma2; // sigma of 2nd Gaussian in missing Ht resolution
    double MHtGaussFr; // relative fraction of 2nd Gaussian
    double HtOffset; // HT offset
    double beamEnergy; // beam energy (Tevatron=980, LHC-1=3500.0) 
    int InputReorder; // flag for input re-order
    int METScanScheme; // MET-scan scheme: 0- use JER; 1- use simple sumEt & missingHt for Njet=0 events in (lep-had) 
    int METresSyst; // switch to turn on/off MET resolution systematics
    bool fUseDefaults; // switch to control defaults: 1== use defaults, 0== don't use defaults (useful for studies) 
    bool fUseTailCleanup; // switch to apply tail clean-up
    bool fUseVerbose; // code to turn ON printouts for debugging
    TVector2 m_metVec;
    TLorentzVector tlv_tmp;
    double m_inputMEtX,m_inputMEtY,m_inputMEtT;
    double m_MEtX, m_MEtY,m_MEtT;
    double m_HtOffset;
    int LFVmode; // flag to determine which LFV decay to be reconstructed: 0=H->e+tau(mu) or 1=H->mu+tau(e) 
    TauTypes::e m_tauTypes;
};
} // namespace DiTauMassTools

#endif
