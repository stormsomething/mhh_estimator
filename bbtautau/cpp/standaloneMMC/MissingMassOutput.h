/*
  Copyright (C) 2002-2019 CERN for the benefit of the ATLAS collaboration
*/

// Class handling the output of the MissingMassCalculator
// author Michael Huebner <michael.huebner@no.spam.cern.ch>

#ifndef DITAUMASSTOOLS_MISSINGMASSOUTPUT_H
#define DITAUMASSTOOLS_MISSINGMASSOUTPUT_H

// local include(s):
#include "HelperFunctions.h"

// ROOT include(s):
#include <TH1.h>

namespace DiTauMassTools{

class MissingMassCalculatorV2;

class MissingMassOutput {

  public:
    MissingMassOutput();
    ~MissingMassOutput();

    int GetFitStatus(); // return fit status
    double GetFittedMass(int fitcode); // returns fitted Mass
    double GetFittedMassErrorUp(int fitcode); // returns upper error on fitted Mass
    double GetFittedMassErrorLow(int fitcode); // returns lower error on fitted Mass
    std::shared_ptr<TH1F> GetMassHistogram(); // return mass histogram
    std::shared_ptr<TH1F> GetMassHistogramNoWeight(); // return mass histogram without weights
    int GetNTrials();// total number of point scanned
    int GetNSuccesses();// total number of point with at least 1 solutions
    int GetNSolutions();// total number of solutions
    double GetSumW(); // sum of weights
    double GetAveSolRMS(); // ave RMS of solutions (for one event)

    double GetRms2Mpv(); // returns RMS/MPV according to histogram method
    TLorentzVector GetNeutrino4vec(int fitcode, int ind); // returns neutrino 4-vec
    double GetFitSignificance(int fitcode); // returns fit significance
    TLorentzVector GetTau4vec(int fitcode, int ind); // returns full tau 4-vec
    TLorentzVector GetResonanceVec(int fitcode); // returns 4-vec for resonance
    TVector2 GetFittedMetVec(int fitcode); // returns 2-vec for fitted MET

    friend class MissingMassCalculatorV2;

  private:
    void ClearOutput(bool fUseVerbose);
    int FitStatus;
    double FitSignificance[MMCFitMethodV2::MAX];
    double FittedMass[MMCFitMethodV2::MAX];
    double FittedMassUpperError[MMCFitMethodV2::MAX];
    double FittedMassLowerError[MMCFitMethodV2::MAX];
    TLorentzVector nuvec1[MMCFitMethodV2::MAX];
    TLorentzVector objvec1[MMCFitMethodV2::MAX];
    TLorentzVector nuvec2[MMCFitMethodV2::MAX];
    TLorentzVector objvec2[MMCFitMethodV2::MAX];
    TLorentzVector totalvec[MMCFitMethodV2::MAX];
    TVector2 FittedMetVec[MMCFitMethodV2::MAX];
    double RMS2MPV;
    std::shared_ptr<TH1F> hMfit_all;
    std::shared_ptr<TH1F> hMfit_allNoWeight;
    int NTrials;
    int NSuccesses;
    int NSolutions;
    double SumW;
    double AveSolRMS;
};
} // namespace DiTauMassTools

#endif
