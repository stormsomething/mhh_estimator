/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

// Class handling the probability calculation of the MissingMassCalculator
// author Michael Huebner <michael.huebner@no.spam.cern.ch>

#ifndef DITAUMASSTOOLS_MISSINGMASSPROB_H
#define DITAUMASSTOOLS_MISSINGMASSPROB_H

// Local include(s):
#include "HelperFunctions.h"
#include "MissingMassInput.h"

// ROOT include(s):
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TLorentzVector.h>

#include <string>

namespace DiTauMassTools{

class MissingMassCalculatorV2;

class MissingMassProb {

  public:
    MissingMassProb(MMCCalibrationSetV2::e aset, std::string m_paramFilePath);
    ~MissingMassProb();

    double apply(MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2, bool constant=false, bool oneTau=false, bool twoTau=false);

    void setParamAngle(const TLorentzVector & tauvec, int tau, int tautype);
    void setParamRatio(int tau, int tautype);
    void setParamNuMass();
    TF1* GetFormulaAngle1() {return formulaAngle1;}
    TF1* GetFormulaAngle2() {return formulaAngle2;}
    TF1* GetFormulaRatio1() {return formulaRatio1;}
    TF1* GetFormulaRatio2() {return formulaRatio2;}
    TF1* GetFormulaNuMass() {return formulaNuMass;}

    void SetAllowUseHT(bool allowUseHT) { allowUseHT=allowUseHT;}
    bool GetAllowUseHT() { return allowUseHT;}

    void SetUseHT(bool val) { UseHT=val; }
    bool GetUseHT() { return UseHT; } // if use HT

    void SetUseTauProbability(bool val) { fUseTauProbability=val; }
    bool GetUseTauProbability() { return fUseTauProbability; }

    void SetUseMnuProbability(bool val) { fUseMnuProbability=val; }
    bool GetUseMnuProbability() { return fUseMnuProbability; }

    void SetUseDphiLL(bool val) {fUseDphiLL = val;}
    bool GetUseDphiLL() { return fUseDphiLL; }

    double MetProbability(MissingMassInput& preparedInput, const double & met1,const  double & met2,const  double & MetSigma1, const double & MetSigma2);
    double dTheta3Dparam(const int & parInd, const int & tau_type, const double & P_tau,const double *par); 

    double dTheta3d_probabilityFast(MissingMassInput& preparedInput, const int & tau_type,const double & dTheta3d,const double & P_tau);
    double myDelThetaHadFunc(double *x, double *par);
    double myDelThetaLepFunc(double *x, double *par);

    double MHtProbability(MissingMassInput& preparedInput, const double & d_mhtX, const double & d_mhtY, const double & mht, 
			       const double & trueMetGuess, const double & mht_offset);
    double MHtProbabilityHH(MissingMassInput& preparedInput, const double & d_mhtX, const double & d_mhtY, const double & mht, 
				 const double & trueMetGuess, const double & mht_offset);

    void MET(MissingMassInput& preparedInput);
    double mEtAndTauProbability(MissingMassInput& preparedInput);
    double MnuProbability(MissingMassInput& preparedInput, double mnu, double binsize);
    double MnuProbability(MissingMassInput& preparedInput, double mnu);
    double TauProbability(MissingMassInput& preparedInput, const int & type1, const TLorentzVector & vis1, const TLorentzVector & nu1, 
			       const int & type2, const TLorentzVector & vis2, const TLorentzVector & nu2);
    double TauProbability(MissingMassInput& preparedInput, const int & type1, const TLorentzVector & vis1, const TLorentzVector & nu1, 
			       const int & type2, const TLorentzVector & vis2, const TLorentzVector & nu2, const double & detmet);
    double TauProbabilityLFV(MissingMassInput& preparedInput, const int & type1, const TLorentzVector & vis1, const TLorentzVector & nu1);

    static double MetProbabilityWrapper(MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2);
    static double mEtAndTauProbabilityWrapper(MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2);
    static double dTheta3d_probabilityFastWrapper( MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2 );
    static double TauProbabilityWrapper( MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2 );
    static double MnuProbabilityWrapper( MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2 );
    static double MnuProbabilityNewWrapper( MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2 );

     static double dTheta3d_probabilityNewWrapper( MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2 );
     static double TauProbabilityNewWrapper( MissingMassProb* prob, MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2 );

    std::list<std::function<double(MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2)>> m_probListConstant;
    std::list<std::function<double(MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2)>> m_probListOneTau;
    std::list<std::function<double(MissingMassInput& preparedInput, const int & tau_type1, const int & tau_type2, const TLorentzVector & tauvec1, const TLorentzVector & tauvec2, const TLorentzVector nuvec1, const TLorentzVector & nuvec2)>> m_probListTwoTau;
  private:
    //SpeedUp static array for efficient access
    static double fit_param[2][3][6][5];
    static double ter_sigma_par[2][10][3];
    //cache quantities for efficient NuPSolution calculation

    TF1 *formulaAngle1 = new TF1("formulaAngle1", "[0]*exp(-[2]*(log((x+[3])/[1]))**2)");
    TF1 *formulaAngle2 = new TF1("formulaAngle2", "[0]*exp(-[2]*(log((x+[3])/[1]))**2)");
    TF1 *formulaRatio1;
    TF1 *formulaRatio2;
    TF1 *formulaRatioLep1 = new TF1("formulaRatio1", "gaus(0)+expo(3)");
    TF1 *formulaRatioLep2 = new TF1("formulaRatio2", "gaus(0)+expo(3)");
    TF1 *formulaRatioHad1 = new TF1("formulaRatio1", "gaus(0)");
    TF1 *formulaRatioHad2 = new TF1("formulaRatio2", "gaus(0)");
    TF1 *formulaNuMass = new TF1("formulaNuMass", "pol6");
    std::vector<TF1*> paramVectorAngle;
    std::vector<TF1*> paramVectorAngleLep;
    std::vector<TF1*> paramVectorRatio;
    std::vector<TF1*> paramVectorRatioLep;
    std::vector<TF1*> paramVectorNuMass;

    std::string m_paramFilePath;
    TFile *fParams;
    MMCCalibrationSetV2::e m_MMCCalibrationSet;

    bool allowUseHT;
    bool UseHT;
    bool fUseTauProbability; // switch to apply TauProbability
    bool fUseMnuProbability; // switch to apply MnuProbability
    bool fUseDphiLL; //for leplep

  // comment by Sasha to keep this for the future
  //inline double AngularProbability(TLorentzVector nu_vec, TLorentzVector vis_vec, int decayType);

  // to be tested if helpful
  //inline double ResonanceProbability(TLorentzVector vec1, TLorentzVector vec2);
};
} // namespace DiTauMassTools

#endif
