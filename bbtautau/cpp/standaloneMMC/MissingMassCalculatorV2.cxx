/*
  Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration
*/

// vim: ts=8 sw=2
/*
  Missing Mass Calculator
*/
//
// to be done : tau 4-vect and type should be data member of MMC.

// if histogram smoothing
//#define SMOOTH
//#define SAVELIKELIHOODHISTO // use with caution: increases run-time by a
// factor of ~10

#include "MissingMassCalculatorV2.h" // this is for RootCore package
#include "MissingMassInput.cxx" // include other cxx for easier compilation
#include "MissingMassOutput.cxx" // include other cxx for easier compilation
#include "MissingMassProb.cxx" // include other cxx for easier compilation
#include "HelperFunctions.cxx" // include other cxx for easier compilation
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
// #include "MissingMassCalculatorV2.h" // this is for standalone
// package

#include <TObject.h>
// SpeedUp committed from revision 163876
#include <TF1.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TMatrixDSym.h>
#include <TObject.h>
#include <TVectorD.h>

using namespace DiTauMassTools;

//______________________________constructor________________________________
MissingMassCalculatorV2::MissingMassCalculatorV2(
    MMCCalibrationSetV2::e aset, std::string m_paramFilePath)
    : randomGen(), Prob(new MissingMassProb(aset, m_paramFilePath)) {
  m_MMCCalibrationSet = aset;
  preparedInput.fUseVerbose = 0;
  preparedInput.beamEnergy = 6500.0; // for now LHC default is sqrt(S)=7 TeV
  Niter_fit1 = 20;
  Niter_fit2 = 30;
  Niter_fit3 = 10;
  NsucStop = -1;
  NiterRandom = -1;        // if the user does not set it to positive value,will be set
                           // in SpaceWalkerInit
  m_NiterRandomLocal = -1; // niterandom which is really used
  // to be used with RMSSTOP  NiterRandom=10000000; // number of random
  // iterations for lh. Multiplied by 10 for ll, divided by 10 for hh (to be
  // optimised)
  //  RMSStop=200;// Stop criteria depending of rms of histogram
  reRunWithBestMET = false;
  RMSStop = -1; // disable

  RndmSeedAltering = 0;    // can be changed to re-compute with different random seed
  dRmax_tau = 0.4;         // changed from 0.2
  Nsigma_METscan = -1;     // number of sigmas for MET scan
  Nsigma_METscan_ll = 3.0; // number of sigmas for MET scan
  Nsigma_METscan_lh = 3.0; // number of sigmas for MET scan
  Nsigma_METscan_hh = 4.0; // number of sigmas for MET scan (4 for hh 2013)

  m_meanbinStop = -1;     // meanbin stopping criterion (-1 if not used)
  m_ProposalTryMEt = -1;  // loop on METproposal disable // FIXME should be cleaner
  m_ProposalTryPhi = -1;  // loop on Phiproposal disable
  m_ProposalTryMnu = -1;  // loop on MNuProposal disable
  m_ProposalTryEtau = -1; // loop on ETauProposal disable

  Prob->SetUseTauProbability(true);  // TauProbability is ON by default DRMERGE comment out for now
  Prob->SetUseMnuProbability(false); // MnuProbability is OFF by default
  Prob->SetUseDphiLL(false);         // added by Tomas Davidek for lep-lep
  dTheta3d_binMin = 0.0025;
  dTheta3d_binMax = 0.02;
  fJERsyst = 0;                 // no JER systematics by default (+/-1: up/down 1 sigma)
  preparedInput.METresSyst = 0; // no MET resolution systematics by default (+/-1: up/down 1 sigma)
  fApplyMassScale = 0;          // don't apply mass scale correction by default
  preparedInput.dataType = 1;   // set to "data" by default
  preparedInput.fUseTailCleanup = 1; // cleanup by default for lep-had Moriond 2012 analysis
  preparedInput.fUseDefaults = 0; // use pre-set defaults for various configurations; if set it to 0
                                  // if need to study various options
  fUseEfficiencyRecovery = 0;     // no re-fit by default
  fUseFloatStopping = 1;          // use floating stopping criterion by default

  preparedInput.METScanScheme = 1; // MET-scan scheme: 0- use JER; 1- use simple sumEt & missingHt
                                   // for Njet=0 events in (lep-had winter 2012)
  //  MnuScanRange=1.777; // range of M(nunu) scan
  MnuScanRange = 1.5;         // better value (sacha)
  preparedInput.LFVmode = -1; // by default consider case of H->mu+tau(->ele)
  preparedInput.ClearInput();

  nCallprobCalculatorV9fast = 0;
  iterTheta3d = 0;
  debugThisIteration = false;

  nsolmax = 4;
  nsolfinalmax = nsolmax * nsolmax;

  nuvecsol1.resize(nsolmax);
  nuvecsol2.resize(nsolmax);
  tauvecsol1.resize(nsolmax);
  tauvecsol2.resize(nsolmax);
  tauvecprob1.resize(nsolmax);
  tauvecprob2.resize(nsolmax);

  m_nsol = 0;
  m_probFinalSolVec.resize(nsolfinalmax);
  m_mtautauFinalSolVec.resize(nsolfinalmax);
  m_nu1FinalSolVec.resize(nsolfinalmax);
  m_nu2FinalSolVec.resize(nsolfinalmax);

  m_nsolOld = 0;
  m_probFinalSolOldVec.resize(nsolfinalmax);
  m_mtautauFinalSolOldVec.resize(nsolfinalmax);
  m_nu1FinalSolOldVec.resize(nsolfinalmax);
  m_nu2FinalSolOldVec.resize(nsolfinalmax);

  float hEmax = 3000.0; // maximum energy (GeV)
  // number of bins
  int hNbins = 1500; // original 2500 for mass, 10000 for P
  // choice of hNbins also related to size of window for fitting (see
  // maxFromHist)

  //--- define histograms for histogram method
  //--- upper limits need to be revisied in the future!!! It may be not enough
  // for some analyses

  fMfit_all = std::make_shared<TH1F>("MMC_h1", "M", hNbins, 0.0,
                                     hEmax); // all solutions
#ifdef SAVELIKELIHOODHISTO
  fMEtP_all = std::make_shared<TH1F>("MEtP_h1", "M", hNbins, -100.0,
                                     100.); // all solutions
  fMEtL_all = std::make_shared<TH1F>("MEtL_h1", "M", hNbins, -100.0,
                                     100.); // all solutions
  fMnu1_all = std::make_shared<TH1F>("Mnu1_h1", "M", hNbins, 0.0,
                                     hEmax); // all solutions
  fMnu2_all = std::make_shared<TH1F>("Mnu2_h1", "M", hNbins, 0.0,
                                     hEmax); // all solutions
  fPhi1_all = std::make_shared<TH1F>("Phi1_h1", "M", hNbins, -10.0,
                                     10.); // all solutions
  fPhi2_all = std::make_shared<TH1F>("Phi2_h1", "M", hNbins, -10.0,
                                     10.);     // all solutions
  fMfit_allGraph = std::make_shared<TGraph>(); // all solutions
#endif
  fMfit_all->Sumw2(); // allow proper error bin calculation. Slightly slower but
                      // completely negligible
  // these histograms are used for the floating stopping criterion
  if (fUseFloatStopping) {
    fMmass_split1 = std::make_shared<TH1F>("mass_h1_1", "M", hNbins, 0.0, hEmax);
    fMEtP_split1 = std::make_shared<TH1F>("MEtP_h1_1", "M", hNbins, -100.0, 100.0);
    fMEtL_split1 = std::make_shared<TH1F>("MEtL_h1_1", "M", hNbins, -100.0, 100.0);
    fMnu1_split1 = std::make_shared<TH1F>("Mnu1_h1_1", "M", hNbins, 0.0, hEmax);
    fMnu2_split1 = std::make_shared<TH1F>("Mnu2_h1_1", "M", hNbins, 0.0, hEmax);
    fPhi1_split1 = std::make_shared<TH1F>("Phi1_h1_1", "M", hNbins, -10.0, 10.0);
    fPhi2_split1 = std::make_shared<TH1F>("Phi2_h1_1", "M", hNbins, -10.0, 10.0);
    fMmass_split2 = std::make_shared<TH1F>("mass_h1_2", "M", hNbins, 0.0, hEmax);
    fMEtP_split2 = std::make_shared<TH1F>("MEtP_h1_2", "M", hNbins, -100.0, 100.0);
    fMEtL_split2 = std::make_shared<TH1F>("MEtL_h1_2", "M", hNbins, -100.0, 100.0);
    fMnu1_split2 = std::make_shared<TH1F>("Mnu1_h1_2", "M", hNbins, 0.0, hEmax);
    fMnu2_split2 = std::make_shared<TH1F>("Mnu2_h1_2", "M", hNbins, 0.0, hEmax);
    fPhi1_split2 = std::make_shared<TH1F>("Phi1_h1_2", "M", hNbins, -10.0, 10.0);
    fPhi2_split2 = std::make_shared<TH1F>("Phi2_h1_2", "M", hNbins, -10.0, 10.0);
  }
#ifdef SAVELIKELIHOODHISTO
  fMEtP_all->Sumw2();
  fMEtL_all->Sumw2();
  fMnu1_all->Sumw2();
  fMnu2_all->Sumw2();
  fPhi1_all->Sumw2();
  fPhi2_all->Sumw2();
#endif
  // these histograms are used for the floating stopping criterion
  if (fUseFloatStopping) {
    fMmass_split1->Sumw2();
    fMEtP_split1->Sumw2();
    fMEtL_split1->Sumw2();
    fMnu1_split1->Sumw2();
    fMnu2_split1->Sumw2();
    fPhi1_split1->Sumw2();
    fPhi2_split1->Sumw2();
    fMmass_split2->Sumw2();
    fMEtP_split2->Sumw2();
    fMEtL_split2->Sumw2();
    fMnu1_split2->Sumw2();
    fMnu2_split2->Sumw2();
    fPhi1_split2->Sumw2();
    fPhi2_split2->Sumw2();
  }

  // histogram without weight. useful for debugging. negligibly slow until now
  fMfit_allNoWeight =
      std::make_shared<TH1F>("MMC_h1NoW", "M no weight", hNbins, 0.0, hEmax); // all solutions

  fPXfit1 = std::make_shared<TH1F>("MMC_h2", "Px1", 4 * hNbins, -hEmax,
                                   hEmax); // Px for tau1
  fPYfit1 = std::make_shared<TH1F>("MMC_h3", "Py1", 4 * hNbins, -hEmax,
                                   hEmax); // Py for tau1
  fPZfit1 = std::make_shared<TH1F>("MMC_h4", "Pz1", 4 * hNbins, -hEmax,
                                   hEmax); // Pz for tau1
  fPXfit2 = std::make_shared<TH1F>("MMC_h5", "Px2", 4 * hNbins, -hEmax,
                                   hEmax); // Px for tau2
  fPYfit2 = std::make_shared<TH1F>("MMC_h6", "Py2", 4 * hNbins, -hEmax,
                                   hEmax); // Py for tau2
  fPZfit2 = std::make_shared<TH1F>("MMC_h7", "Pz2", 4 * hNbins, -hEmax,
                                   hEmax); // Pz for tau2

  fMfit_all->SetDirectory(0);
#ifdef SAVELIKELIHOODHISTO
  fMEtP_all->SetDirectory(0);
  fMEtL_all->SetDirectory(0);
  fMnu1_all->SetDirectory(0);
  fMnu2_all->SetDirectory(0);
  fPhi1_all->SetDirectory(0);
  fPhi2_all->SetDirectory(0);
#endif
  fMfit_allNoWeight->SetDirectory(0);
  fPXfit1->SetDirectory(0);
  fPYfit1->SetDirectory(0);
  fPZfit1->SetDirectory(0);
  fPXfit2->SetDirectory(0);
  fPYfit2->SetDirectory(0);
  fPZfit2->SetDirectory(0);

  // these histograms are used for the floating stopping criterion
  if (fUseFloatStopping) {
    fMmass_split1->SetDirectory(0);
    fMEtP_split1->SetDirectory(0);
    fMEtL_split1->SetDirectory(0);
    fMnu1_split1->SetDirectory(0);
    fMnu2_split1->SetDirectory(0);
    fPhi1_split1->SetDirectory(0);
    fPhi2_split1->SetDirectory(0);
    fMmass_split2->SetDirectory(0);
    fMEtP_split2->SetDirectory(0);
    fMEtL_split2->SetDirectory(0);
    fMnu1_split2->SetDirectory(0);
    fMnu2_split2->SetDirectory(0);
    fPhi1_split2->SetDirectory(0);
    fPhi2_split2->SetDirectory(0);
  }

  // max hist fitting function
  m_fFitting =
      new TF1("MMC_maxFitting", this, &MissingMassCalculatorV2::maxFitting, 0., hEmax, 3);
  // Sets initial parameter names
  m_fFitting->SetParNames("Max", "Mean", "InvWidth2");

  if (preparedInput.fUseVerbose == 1) {
    gDirectory->pwd();
    gDirectory->ls();
  }

  if (preparedInput.fUseVerbose == 1) {
    gDirectory->pwd();
    gDirectory->ls();
  }
}

MissingMassCalculatorV2::~MissingMassCalculatorV2() { delete Prob; }



int MissingMassCalculatorV2::MMCforBoom(const float tau_0_pt, const float tau_0_eta, const float tau_0_phi, const float tau_0_m,
	       const float tau_1_pt, const float tau_1_eta, const float tau_1_phi, const float tau_1_m,
	       const int mmcType1, const int mmcType2, const float met_x, const float met_y, const float sumet, const int &njets, const int evtnum)
{
  // Format values into 4-vectors
  TLorentzVector tlvTau1, tlvTau2;
  tlvTau1.SetPtEtaPhiM(tau_0_pt, tau_0_eta, tau_0_phi, tau_0_m);
  tlvTau2.SetPtEtaPhiM(tau_1_pt, tau_1_eta, tau_1_phi, tau_1_m);
  
  // Set event number
  SetEventNumber(evtnum);

  // Call main function
  // Reminder: we actually want it to return the actual MMC, but we can interface that later
  return RunMissingMassCalculator(tlvTau1, tlvTau2, mmcType1, mmcType2, met_x, met_y, sumet, njets);

}

//_____________________________________________________________________________
// Main Method to run MissingMassCalculator
int MissingMassCalculatorV2::RunMissingMassCalculator(const TLorentzVector tlvTau1,
						      const TLorentzVector tlvTau2,
						      const int mmcType1,
						      const int mmcType2,
						      const float met_x,
						      const float met_y,
						      const float sumet,
						      const int &njets) {
  reRunWithBestMET = false;

  OutputInfo.ClearOutput(preparedInput.fUseVerbose);
  if (preparedInput.fUseVerbose == 1) {
    Info("DiTauMassTools", "------------- Raw Input for MissingMassCalculator --------------");
  }
  FinalizeSettings(tlvTau1, tlvTau2, mmcType1, mmcType2, met_x, met_y, sumet,  njets); // rawInput, preparedInput );
  Prob->MET(preparedInput);
  if (preparedInput.fUseVerbose == 1) {
    Info("DiTauMassTools", "------------- Prepared Input for MissingMassCalculator--------------");
    preparedInput.PrintInputInfo();
  }

  if (preparedInput.LFVmode <= 0) {
    // remove argument DiTauMassCalculatorV9Walk work directly on preparedInput
    OutputInfo.FitStatus = DitauMassCalculatorV9walk();

    // re-running MMC for on failed events
    if (fUseEfficiencyRecovery == 1 && OutputInfo.FitStatus != 1) {
      // most events where MMC failed happened to have dPhi>2.9. Run re-fit only
      // on these events
      if (preparedInput.DelPhiTT > 2.9) {
        //              preparedInput.MetVec.Set(-(preparedInput.vistau1+preparedInput.vistau2).Px(),-(preparedInput.vistau1+preparedInput.vistau2).Py());
        //              // replace MET by MPT

        TVector2 dummy_met(-(preparedInput.vistau1 + preparedInput.vistau2).Px(),
                           -(preparedInput.vistau1 + preparedInput.vistau2).Py());
        preparedInput.METcovphi = dummy_met.Phi();
        double dummy_METres =
            sqrt(pow(preparedInput.METsigmaL, 2) + pow(preparedInput.METsigmaP, 2));
        preparedInput.METsigmaL =
            dummy_METres * fabs(cos(dummy_met.Phi() - preparedInput.MetVec.Phi()));
        preparedInput.METsigmaP =
            dummy_METres * fabs(sin(dummy_met.Phi() - preparedInput.MetVec.Phi()));
        if (preparedInput.METsigmaP < 5.0)
          preparedInput.METsigmaP = 5.0;
        Nsigma_METscan_lh = 6.0; // increase range of MET scan
        Nsigma_METscan_hh = 6.0; // increase range of MET scan

        OutputInfo.ClearOutput(preparedInput.fUseVerbose);  // clear output stuff before re-running
        OutputInfo.FitStatus = DitauMassCalculatorV9walk(); // run MMC again
      }
    }

  }

  // running MMC in LFV mode for reconstructing mass of X->lep+tau
  else {
    if (preparedInput.fUseVerbose == 1) {
      Info("DiTauMassTools", "Calling DitauMassCalculatorV9lfv");
    }
    OutputInfo.FitStatus = DitauMassCalculatorV9lfv();
  }
#ifdef SAVELIKELIHOODHISTO
  TFile *outFile = TFile::Open("MMC_likelihoods.root", "UPDATE");
  outFile->cd();
  auto path = std::to_string(m_eventNumber);
  if (!outFile->GetDirectory(path.c_str()))
    outFile->mkdir(path.c_str());
  outFile->cd(path.c_str());
  fMfit_all->Write(fMfit_all->GetName(), TObject::kOverwrite);
  fMEtP_all->Write(fMEtP_all->GetName(), TObject::kOverwrite);
  fMEtL_all->Write(fMEtL_all->GetName(), TObject::kOverwrite);
  fMnu1_all->Write(fMnu1_all->GetName(), TObject::kOverwrite);
  fMnu2_all->Write(fMnu2_all->GetName(), TObject::kOverwrite);
  fPhi1_all->Write(fPhi1_all->GetName(), TObject::kOverwrite);
  fPhi2_all->Write(fPhi2_all->GetName(), TObject::kOverwrite);
  fMfit_allNoWeight->Write(fMfit_allNoWeight->GetName(), TObject::kOverwrite);
  fMfit_allGraph->Write("Graph", TObject::kOverwrite);
  TH1D *nosol = new TH1D("nosol", "nosol", 7, 0, 7);
  nosol->SetBinContent(1, m_testptn1);
  nosol->SetBinContent(2, m_testptn2);
  nosol->SetBinContent(3, m_testdiscri1);
  nosol->SetBinContent(4, m_testdiscri2);
  nosol->SetBinContent(5, m_nosol1);
  nosol->SetBinContent(6, m_nosol1);
  nosol->SetBinContent(7, m_iterNuPV3);
  nosol->Write(nosol->GetName(), TObject::kOverwrite);
  outFile->Write();
  outFile->Close();
#endif
  DoOutputInfo();
  PrintResults();
  preparedInput.ClearInput();
  return 1;
}

//-------- clearing ditau container
void MissingMassCalculatorV2::ClearDitauStuff(DitauStuff &fStuff) {
  fStuff.Mditau_best = 0.0;
  fStuff.Sign_best = 1.0E6;
  fStuff.nutau1 = TLorentzVector(0., 0., 0., 0.);
  fStuff.nutau2 = TLorentzVector(0., 0., 0., 0.);
  fStuff.vistau1 = TLorentzVector(0., 0., 0., 0.);
  fStuff.vistau2 = TLorentzVector(0., 0., 0., 0.);
  fStuff.RMSoverMPV = 0.0;

  return;
}

//---------------------------- Accessors to output parameters
//------------------------
// finalizes output information
void MissingMassCalculatorV2::DoOutputInfo() {
  if (OutputInfo.FitStatus > 0) {
    if (preparedInput.fUseVerbose == 1) {
      Info("DiTauMassTools", "Retrieving output from fDitauStuffFit");
    }
    // MAXW method : get from fDittauStuffFit
    OutputInfo.FitSignificance[MMCFitMethodV2::MAXW] = fDitauStuffFit.Sign_best;
    OutputInfo.FittedMass[MMCFitMethodV2::MAXW] = fDitauStuffFit.Mditau_best;
    double q1 = (1. - 0.68) / 2.;
    double q2 = 1. - q1;
    double xq[2], yq[2];
    xq[0] = q1;
    xq[1] = q2;
    fMfit_all->GetQuantiles(2, yq, xq);
    OutputInfo.FittedMassLowerError[MMCFitMethodV2::MAXW] = yq[0];
    OutputInfo.FittedMassUpperError[MMCFitMethodV2::MAXW] = yq[1];
    OutputInfo.nuvec1[MMCFitMethodV2::MAXW] = fDitauStuffFit.nutau1;
    OutputInfo.objvec1[MMCFitMethodV2::MAXW] =
        fDitauStuffFit.vistau1 + fDitauStuffFit.nutau1;
    OutputInfo.nuvec2[MMCFitMethodV2::MAXW] = fDitauStuffFit.nutau2;
    OutputInfo.objvec2[MMCFitMethodV2::MAXW] =
        fDitauStuffFit.vistau2 + fDitauStuffFit.nutau2;
    OutputInfo.totalvec[MMCFitMethodV2::MAXW] =
        OutputInfo.objvec1[MMCFitMethodV2::MAXW] +
        OutputInfo.objvec2[MMCFitMethodV2::MAXW];
    TVector2 metmaxw(OutputInfo.nuvec1[MMCFitMethodV2::MAXW].Px() +
                         OutputInfo.nuvec2[MMCFitMethodV2::MAXW].Px(),
                     OutputInfo.nuvec1[MMCFitMethodV2::MAXW].Py() +
                         OutputInfo.nuvec2[MMCFitMethodV2::MAXW].Py());
    OutputInfo.FittedMetVec[MMCFitMethodV2::MAXW] = metmaxw;

    // MLM method : can only get MMC, rest is dummy
    double scale = MassScale(MMCFitMethodV2::MAXW, fDitauStuffHisto.Mditau_best,
                             preparedInput.type_visTau1,
                             preparedInput.type_visTau2); // only for histo method for now. In
                                                          // practice disabled by default

    OutputInfo.FittedMass[MMCFitMethodV2::MLM] = scale * fDitauStuffHisto.Mditau_best;
    OutputInfo.FittedMassLowerError[MMCFitMethodV2::MLM] = yq[0];
    OutputInfo.FittedMassUpperError[MMCFitMethodV2::MLM] = yq[1];

    TLorentzVector tlvdummy(0., 0., 0., 0.);
    TVector2 metdummy(0., 0.);
    OutputInfo.FitSignificance[MMCFitMethodV2::MLM] = -1.;
    OutputInfo.nuvec1[MMCFitMethodV2::MLM] = tlvdummy;
    OutputInfo.objvec1[MMCFitMethodV2::MLM] = tlvdummy;
    OutputInfo.nuvec2[MMCFitMethodV2::MLM] = tlvdummy;
    OutputInfo.objvec2[MMCFitMethodV2::MLM] = tlvdummy;
    OutputInfo.totalvec[MMCFitMethodV2::MLM] = tlvdummy;
    OutputInfo.FittedMetVec[MMCFitMethodV2::MLM] = metdummy;

    // MLNU3P method : get from fDittauStuffHisto 4 momentum
    OutputInfo.nuvec1[MMCFitMethodV2::MLNU3P] = fDitauStuffHisto.nutau1;
    OutputInfo.objvec1[MMCFitMethodV2::MLNU3P] =
        fDitauStuffHisto.vistau1 + fDitauStuffHisto.nutau1;
    OutputInfo.nuvec2[MMCFitMethodV2::MLNU3P] = fDitauStuffHisto.nutau2;
    OutputInfo.objvec2[MMCFitMethodV2::MLNU3P] =
        fDitauStuffHisto.vistau2 + fDitauStuffHisto.nutau2;
    OutputInfo.totalvec[MMCFitMethodV2::MLNU3P] =
        OutputInfo.objvec1[MMCFitMethodV2::MLNU3P] +
        OutputInfo.objvec2[MMCFitMethodV2::MLNU3P];
    OutputInfo.FittedMass[MMCFitMethodV2::MLNU3P] =
        OutputInfo.totalvec[MMCFitMethodV2::MLNU3P].M();
    OutputInfo.FittedMassUpperError[MMCFitMethodV2::MLNU3P] = 0.;
    OutputInfo.FittedMassLowerError[MMCFitMethodV2::MLNU3P] = 0.;

    TVector2 metmlnu3p(OutputInfo.nuvec1[MMCFitMethodV2::MLNU3P].Px() +
                           OutputInfo.nuvec2[MMCFitMethodV2::MLNU3P].Px(),
                       OutputInfo.nuvec1[MMCFitMethodV2::MLNU3P].Py() +
                           OutputInfo.nuvec2[MMCFitMethodV2::MLNU3P].Py());
    OutputInfo.FittedMetVec[MMCFitMethodV2::MLNU3P] = metmlnu3p;

    OutputInfo.RMS2MPV = fDitauStuffHisto.RMSoverMPV;
  }

  OutputInfo.hMfit_all = fMfit_all;
  OutputInfo.hMfit_allNoWeight = fMfit_allNoWeight;
  OutputInfo.NSolutions = fMfit_all->GetEntries();
  OutputInfo.SumW = fMfit_all->GetSumOfWeights();

  //----------------- Check if input was re-ordered in FinalizeInputStuff() and
  // restore the original order if needed
  if (preparedInput.InputReorder == 1) {
    TLorentzVector dummy_vec1(0.0, 0.0, 0.0, 0.0);
    TLorentzVector dummy_vec2(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < 3; i++) {
      // re-ordering neutrinos
      dummy_vec1 = OutputInfo.nuvec1[i];
      dummy_vec2 = OutputInfo.nuvec2[i];
      OutputInfo.nuvec1[i] = dummy_vec2;
      OutputInfo.nuvec2[i] = dummy_vec1;
      // re-ordering tau's
      dummy_vec1 = OutputInfo.objvec1[i];
      dummy_vec2 = OutputInfo.objvec2[i];
      OutputInfo.objvec1[i] = dummy_vec2;
      OutputInfo.objvec2[i] = dummy_vec1;
    }
  }

  return;
}

// Printout of final results
void MissingMassCalculatorV2::PrintOtherInput() {
  if (preparedInput.fUseVerbose != 1)
    return;

  Info("DiTauMassTools",
       ".........................Other input.....................................");
  Info("DiTauMassTools", "%s",
       ("Beam energy =" + std::to_string(preparedInput.beamEnergy) +
        "  sqrt(S) for collisions =" + std::to_string(2.0 * preparedInput.beamEnergy))
           .c_str());
  Info("DiTauMassTools", "%s",
       ("CalibrationSet " + MMCCalibrationSetV2::name[m_MMCCalibrationSet])
           .c_str());
  Info("DiTauMassTools", "%s",
       ("LFV mode " + std::to_string(preparedInput.LFVmode) + " seed=" + std::to_string(m_seed))
           .c_str());
  Info("DiTauMassTools", "%s", ("usetauProbability=" + std::to_string(Prob->GetUseTauProbability()) +
                          " useTailCleanup=" + std::to_string(preparedInput.fUseTailCleanup))
                             .c_str());

  if (preparedInput.InputReorder != 0) {
    Info("DiTauMassTools",
         "tau1 and tau2 were internally swapped (visible on prepared input printout)");
  } else {
    Info("DiTauMassTools", "tau1 and tau2 were NOT internally swapped");
  }

  Info("DiTauMassTools", "%s",
       (" MEtLMin=" + std::to_string(m_MEtLMin) + " MEtLMax=" + std::to_string(m_MEtLMax)).c_str());
  Info("DiTauMassTools", "%s",
       (" MEtPMin=" + std::to_string(m_MEtPMin) + " MEtPMax=" + std::to_string(m_MEtPMax)).c_str());
  Info("DiTauMassTools", "%s",
       (" Phi1Min=" + std::to_string(m_Phi1Min) + " Phi1Max=" + std::to_string(m_Phi1Max)).c_str());
  Info("DiTauMassTools", "%s",
       (" Phi2Min=" + std::to_string(m_Phi2Min) + " Phi2Max=" + std::to_string(m_Phi2Max)).c_str());
  Info("DiTauMassTools", "%s",
       (" Mnu1Min=" + std::to_string(m_Mnu1Min) + " Mnu1Max=" + std::to_string(m_Mnu1Max)).c_str());
  Info("DiTauMassTools", "%s",
       (" Mnu2Min=" + std::to_string(m_Mnu2Min) + " Mnu2Max=" + std::to_string(m_Mnu2Max)).c_str());
}

// Printout of final results
void MissingMassCalculatorV2::PrintResults() {

  if (preparedInput.fUseVerbose != 1)
    return;

  const TLorentzVector *origVisTau1 = 0;
  const TLorentzVector *origVisTau2 = 0;

  if (preparedInput.InputReorder == 0) {
    origVisTau1 = &preparedInput.vistau1;
    origVisTau2 = &preparedInput.vistau2;
  } else // input order was flipped
  {
    origVisTau1 = &preparedInput.vistau2;
    origVisTau2 = &preparedInput.vistau1;
  }

  PrintOtherInput();

  Info("DiTauMassTools",
       "------------- Printing Final Results for MissingMassCalculator --------------");
  Info("DiTauMassTools",
       ".............................................................................");
  Info("DiTauMassTools", "%s", ("Fit status=" + std::to_string(OutputInfo.FitStatus)).c_str());

  for (int imeth = 0; imeth < MMCFitMethodV2::MAX; ++imeth) {
    Info("DiTauMassTools", "%s",
         ("___  Results for " + MMCFitMethodV2::name[imeth] + "Method ___")
             .c_str());
    Info("DiTauMassTools", "%s",
         (" signif=" + std::to_string(OutputInfo.FitSignificance[imeth])).c_str());
    Info("DiTauMassTools", "%s", (" mass=" + std::to_string(OutputInfo.FittedMass[imeth])).c_str());
    Info("DiTauMassTools", "%s", (" rms/mpv=" + std::to_string(OutputInfo.RMS2MPV)).c_str());

    if (imeth == MMCFitMethodV2::MLM) {
      Info("DiTauMassTools", " no 4-momentum or MET from this method ");
      continue;
    }

    if (OutputInfo.FitStatus <= 0) {
      Info("DiTauMassTools", " fit failed ");
    }

    const TLorentzVector &tlvnu1 = OutputInfo.nuvec1[imeth];
    const TLorentzVector &tlvnu2 = OutputInfo.nuvec2[imeth];
    const TLorentzVector &tlvo1 = OutputInfo.objvec1[imeth];
    const TLorentzVector &tlvo2 = OutputInfo.objvec2[imeth];
    const TVector2 &tvmet = OutputInfo.FittedMetVec[imeth];

    Info("DiTauMassTools", "%s",
         (" Neutrino-1: P=" + std::to_string(tlvnu1.P()) + "  Pt=" + std::to_string(tlvnu1.Pt()) +
          "  Eta=" + std::to_string(tlvnu1.Eta()) + "  Phi=" + std::to_string(tlvnu1.Phi()) +
          "  M=" + std::to_string(tlvnu1.M()) + " Px=" + std::to_string(tlvnu1.Px()) +
          " Py=" + std::to_string(tlvnu1.Py()) + " Pz=" + std::to_string(tlvnu1.Pz()))
             .c_str());
    Info("DiTauMassTools", "%s",
         (" Neutrino-2: P=" + std::to_string(tlvnu2.P()) + "  Pt=" + std::to_string(tlvnu2.Pt()) +
          "  Eta=" + std::to_string(tlvnu2.Eta()) + "  Phi=" + std::to_string(tlvnu2.Phi()) +
          "  M=" + std::to_string(tlvnu2.M()) + " Px=" + std::to_string(tlvnu2.Px()) +
          " Py=" + std::to_string(tlvnu2.Py()) + " Pz=" + std::to_string(tlvnu2.Pz()))
             .c_str());
    Info("DiTauMassTools", "%s",
         (" Tau-1: P=" + std::to_string(tlvo1.P()) + "  Pt=" + std::to_string(tlvo1.Pt()) +
          "  Eta=" + std::to_string(tlvo1.Eta()) + "  Phi=" + std::to_string(tlvo1.Phi()) +
          "  M=" + std::to_string(tlvo1.M()) + " Px=" + std::to_string(tlvo1.Px()) +
          " Py=" + std::to_string(tlvo1.Py()) + " Pz=" + std::to_string(tlvo1.Pz()))
             .c_str());
    Info("DiTauMassTools", "%s",
         (" Tau-2: P=" + std::to_string(tlvo2.P()) + "  Pt=" + std::to_string(tlvo2.Pt()) +
          "  Eta=" + std::to_string(tlvo2.Eta()) + "  Phi=" + std::to_string(tlvo2.Phi()) +
          "  M=" + std::to_string(tlvo2.M()) + " Px=" + std::to_string(tlvo2.Px()) +
          " Py=" + std::to_string(tlvo2.Py()) + " Pz=" + std::to_string(tlvo2.Pz()))
             .c_str());

    Info("DiTauMassTools", "%s",
         (" dR(nu1-visTau1)=" + std::to_string(tlvnu1.DeltaR(*origVisTau1))).c_str());
    Info("DiTauMassTools", "%s",
         (" dR(nu2-visTau2)=" + std::to_string(tlvnu2.DeltaR(*origVisTau2))).c_str());

    Info("DiTauMassTools", "%s",
         (" Fitted MET =" + std::to_string(tvmet.Mod()) + "  Phi=" + std::to_string(tlvnu1.Phi()) +
          " Px=" + std::to_string(tvmet.Px()) + " Py=" + std::to_string(tvmet.Py()))
             .c_str());

    Info("DiTauMassTools", "%s", (" Resonance: P=" + std::to_string(OutputInfo.totalvec[imeth].P()) +
                            "  Pt=" + std::to_string(OutputInfo.totalvec[imeth].Pt()) +
                            " Eta=" + std::to_string(OutputInfo.totalvec[imeth].Eta()) +
                            " Phi=" + std::to_string(OutputInfo.totalvec[imeth].Phi()) +
                            " M=" + std::to_string(OutputInfo.totalvec[imeth].M()) +
                            " Px=" + std::to_string(OutputInfo.totalvec[imeth].Px()) +
                            " Py=" + std::to_string(OutputInfo.totalvec[imeth].Py()) +
                            " Pz=" + std::to_string(OutputInfo.totalvec[imeth].Pz()))
                               .c_str());
  }

  return;
}

// returns P1, P2, and theta1 & theta2 solutions
// This compute the nu1 nu2 solution in the most efficient way. Wrt to
// NuPsolutionV2, the output nu1 nu2 4-vector have non zero mass (if relevant).
// It is not optimised for grid running so much less caching is done (which
// makes it more readable). Only quantities fixed within an event are cached.
// relies on a number of these variables to be initialised before the loop.

int MissingMassCalculatorV2::NuPsolutionV3(const double &mNu1, const double &mNu2,
                                                    const double &phi1, const double &phi2,
                                                    int &nsol1, int &nsol2) {

  // Pv1, Pv2 : visible tau decay product momentum
  // Pn1 Pn2 : neutrino momentum
  // phi1, phi2 : neutrino azymutal angles
  // PTmiss2=PTmissy Cos[phi2] - PTmissx Sin[phi2]
  // PTmiss2cscdphi=PTmiss2/Sin[phi1-phi2]
  // Pv1proj=Pv1x Cos[phi1] + Pv1y Sin[phi1]
  // M2noma1=Mtau^2-Mv1^2-Mn1^2
  // ETv1^2=Ev1^2-Pv1z^2

  // discriminant : 16 Ev1^2 (M2noma1^2 + 4 M2noma1 PTmiss2cscdphi Pv1proj -  4
  // (ETv1^2 (Mn1^2 + PTmiss2cscdphi^2) - PTmiss2cscdphi^2 Pv1proj^2))
  // two solutions for epsilon = +/-1
  // Pn1z=(1/(2 ETv1^2))(epsilon Ev1 Sqrt[ M2noma1^2 + 4 M2noma1 PTmiss2cscdphi
  // Pv1proj - 4 (ETv1^2 (Mn1^2 + qPTmiss2cscdphi^2) - PTmiss2cscdphi^2
  // Pv1proj^2)] + M2noma1 Pv1z + 2 PTmiss2cscdphi Pv1proj Pv1z)
  // with conditions:  M2noma1 + 2 PTmiss2cscdphi Pv1proj + 2 Pn1z Pv1z > 0
  // PTn1 -> PTmiss2 Csc[phi1 - phi2]

  // if initialisation precompute some quantities
  int solution_code = 0; // 0 with no solution, 1 with solution
  nsol1 = 0;
  nsol2 = 0;

  // Variables used to test PTn1 and PTn2 > 0

  const double &pTmissx = preparedInput.m_MEtX;
  const double &pTmissy = preparedInput.m_MEtY;

  fastSinCos(phi2, m_sinPhi2, m_cosPhi2);
  double pTmiss2 = pTmissy * m_cosPhi2 - pTmissx * m_sinPhi2;

  int dPhiSign = 0;
  dPhiSign = fixPhiRange(phi1 - phi2) > 0 ? +1 : -1;

  // Test if PTn1 and PTn2 > 0. Then MET vector is between the two neutrino
  // vector

  if (pTmiss2 * dPhiSign < 0) {
    ++m_testptn1;
    return solution_code;
  }

  fastSinCos(phi1, m_sinPhi1, m_cosPhi1);
  double pTmiss1 = pTmissy * m_cosPhi1 - pTmissx * m_sinPhi1;

  if (pTmiss1 * (-dPhiSign) < 0) {
    ++m_testptn2;
    return solution_code;
  }

  // Variables used to calculate discri1

  double m2Vis1 = m_tauVec1M * m_tauVec1M;
  m_ET2v1 = std::pow(m_tauVec1E, 2) - std::pow(m_tauVec1Pz, 2);
  m_m2Nu1 = mNu1 * mNu1;
  double m2noma1 = m_mTau2 - m_m2Nu1 - m2Vis1;
  double m4noma1 = m2noma1 * m2noma1;
  double pv1proj = m_tauVec1Px * m_cosPhi1 + m_tauVec1Py * m_sinPhi1;
  double p2v1proj = std::pow(pv1proj, 2);
  double sinDPhi2 = m_cosPhi2 * m_sinPhi1 - m_sinPhi2 * m_cosPhi1; // sin(Phi1-Phi2)
  double pTmiss2CscDPhi = pTmiss2 / sinDPhi2;
  double &pTn1 = pTmiss2CscDPhi;
  double pT2miss2CscDPhi = pTmiss2CscDPhi * pTmiss2CscDPhi;

  // Test on discri1
  const double discri1 = m4noma1 + 4 * m2noma1 * pTmiss2CscDPhi * pv1proj -
                         4 * (m_ET2v1 * (m_m2Nu1 + pT2miss2CscDPhi) - (pT2miss2CscDPhi * p2v1proj));

  if (discri1 < 0) // discriminant negative -> no solution
  {
    ++m_testdiscri1;
    return solution_code;
  }

  // Variables used to calculate discri2
  double m2Vis2 = m_tauVec2M * m_tauVec2M;
  m_ET2v2 = std::pow(m_tauVec2E, 2) - std::pow(m_tauVec2Pz, 2);
  m_m2Nu2 = mNu2 * mNu2;
  double m2noma2 = m_mTau2 - m_m2Nu2 - m2Vis2;
  double m4noma2 = m2noma2 * m2noma2;
  double pv2proj = m_tauVec2Px * m_cosPhi2 + m_tauVec2Py * m_sinPhi2;
  double p2v2proj = std::pow(pv2proj, 2);
  double sinDPhi1 = -sinDPhi2;
  double pTmiss1CscDPhi = pTmiss1 / sinDPhi1;
  double &pTn2 = pTmiss1CscDPhi;
  double pT2miss1CscDPhi = pTmiss1CscDPhi * pTmiss1CscDPhi;

  const double discri2 = m4noma2 + 4 * m2noma2 * pTmiss1CscDPhi * pv2proj -
                         4 * (m_ET2v2 * (m_m2Nu2 + pT2miss1CscDPhi) - (pT2miss1CscDPhi * p2v2proj));

  if (discri2 < 0) // discriminant negative -> no solution
  {
    ++m_testdiscri2;
    return solution_code;
  }

  // this should be done only once we know there are solutions for nu2
  m_E2v1 = m_tauVec1Pz * m_tauVec1Pz + m_ET2v1;
  m_Ev1 = sqrt(m_E2v1);
  double sqdiscri1 = sqrt(discri1);
  double first1 =
      (m2noma1 * m_tauVec1Pz + 2 * pTmiss2CscDPhi * pv1proj * m_tauVec1Pz) / (2 * m_ET2v1);
  double second1 = sqdiscri1 * m_Ev1 / (2 * m_ET2v1);

  // first solution
  double pn1Z = first1 + second1;

  if (m2noma1 + 2 * pTmiss2CscDPhi * pv1proj + 2 * pn1Z * m_tauVec1Pz >
      0) // Condition for solution to exist
  {
    nuvecsol1[nsol1].SetPxPyPzE(pTn1 * m_cosPhi1, pTn1 * m_sinPhi1, pn1Z,
                                sqrt(std::pow(pTn1, 2) + std::pow(pn1Z, 2) + m_m2Nu1));

    ++nsol1;
  }

  pn1Z = first1 - second1;

  if (m2noma1 + 2 * pTmiss2CscDPhi * pv1proj + 2 * pn1Z * m_tauVec1Pz >
      0) // Condition for solution to exist
  {

    nuvecsol1[nsol1].SetPxPyPzE(pTn1 * m_cosPhi1, pTn1 * m_sinPhi1, pn1Z,
                                sqrt(std::pow(pTn1, 2) + std::pow(pn1Z, 2) + m_m2Nu1));

    ++nsol1;
  }

  if (nsol1 == 0) {
    ++m_nosol1;
    return solution_code;
  }

  m_E2v2 = m_tauVec2Pz * m_tauVec2Pz + m_ET2v2;
  m_Ev2 = sqrt(m_E2v2);
  double sqdiscri2 = sqrt(discri2);
  double first2 =
      (m2noma2 * m_tauVec2Pz + 2 * pTmiss1CscDPhi * pv2proj * m_tauVec2Pz) / (2 * m_ET2v2);
  double second2 = sqdiscri2 * m_Ev2 / (2 * m_ET2v2);

  // second solution
  double pn2Z = first2 + second2;

  if (m2noma2 + 2 * pTmiss1CscDPhi * pv2proj + 2 * pn2Z * m_tauVec2Pz >
      0) // Condition for solution to exist
  {
    nuvecsol2[nsol2].SetPxPyPzE(pTn2 * m_cosPhi2, pTn2 * m_sinPhi2, pn2Z,
                                sqrt(std::pow(pTn2, 2) + std::pow(pn2Z, 2) + m_m2Nu2));

    ++nsol2;
  }

  pn2Z = first2 - second2;
  ;

  if (m2noma2 + 2 * pTmiss1CscDPhi * pv2proj + 2 * pn2Z * m_tauVec2Pz >
      0) // Condition for solution to exist
  {
    nuvecsol2[nsol2].SetPxPyPzE(pTn2 * m_cosPhi2, pTn2 * m_sinPhi2, pn2Z,
                                sqrt(std::pow(pTn2, 2) + std::pow(pn2Z, 2) + m_m2Nu2));

    ++nsol2;
  }

  if (nsol2 == 0) {
    ++m_nosol2;
    return solution_code;
  }

  // Verification if solution exist

  solution_code = 1;
  ++m_iterNuPV3;

  // double check solutions from time to time
  if (m_iterNuPV3 % 1000 == 1) {
    double pnux = nuvecsol1[0].Px() + nuvecsol2[0].Px();
    double pnuy = nuvecsol1[0].Py() + nuvecsol2[0].Py();
    double mtau1plus = (nuvecsol1[0] + m_tauVec1).M();
    double mtau1moins = (nuvecsol1[1] + m_tauVec1).M();
    double mtau2plus = (nuvecsol2[0] + m_tauVec2).M();
    double mtau2moins = (nuvecsol2[1] + m_tauVec2).M();
    if (std::abs(pnux - pTmissx) > 0.001 || std::abs(pnuy - pTmissy) > 0.001) {
      Info("DiTauMassTools", "%s", ("NuPsolutionV3 ERROR Pnux-Met.X or Pnuy-Met.Y > 0.001 : " +
                              std::to_string(pnux - pTmissx) + " and " +
                              std::to_string(pnuy - pTmissx) + " " + "Invalid solutions")
                                 .c_str());
    }
    if (std::abs(mtau1plus - m_mTau) > 0.001 || std::abs(mtau1moins - m_mTau) > 0.001 ||
        std::abs(mtau2plus - m_mTau) > 0.001 || std::abs(mtau2moins - m_mTau) > 0.001) {
      Info("DiTauMassTools", "%s", ("NuPsolutionV3 ERROR tau mass not recovered : " +
                              std::to_string(mtau1plus) + " " + std::to_string(mtau1moins) + " " +
                              std::to_string(mtau2plus) + " " + std::to_string(mtau2moins))
                                 .c_str());
    }
  }

  return solution_code;
}

// returns solution for Lepton Flavor Violating X->lep+tau studies
int MissingMassCalculatorV2::NuPsolutionLFV(const TVector2 &met_vec,
                                                     const TLorentzVector &tau, const double &m_nu,
                                                     std::vector<TLorentzVector> &nu_vec) {
  int solution_code = 0; // 0 with no solution, 1 with solution

  nu_vec.clear();
  TLorentzVector nu(0.0, 0.0, 0.0, 0.0);
  TLorentzVector nu2(0.0, 0.0, 0.0, 0.0);
  nu.SetXYZM(met_vec.Px(), met_vec.Py(), 0.0, m_nu);
  nu2.SetXYZM(met_vec.Px(), met_vec.Py(), 0.0, m_nu);

  const double Mtau = 1.777;
  //   double msq = (Mtau*Mtau-tau.M()*tau.M())/2;
  double msq = (Mtau * Mtau - tau.M() * tau.M() - m_nu * m_nu) /
               2; // to take into account the fact that 2-nu systema has mass
  double gamma = nu.Px() * nu.Px() + nu.Py() * nu.Py();
  double beta = tau.Px() * nu.Px() + tau.Py() * nu.Py() + msq;
  double a = tau.E() * tau.E() - tau.Pz() * tau.Pz();
  double b = -2 * tau.Pz() * beta;
  double c = tau.E() * tau.E() * gamma - beta * beta;
  if ((b * b - 4 * a * c) < 0)
    return solution_code; // no solution found
  else
    solution_code = 2;
  double pvz1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
  double pvz2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
  nu.SetXYZM(met_vec.Px(), met_vec.Py(), pvz1, m_nu);
  nu2.SetXYZM(met_vec.Px(), met_vec.Py(), pvz2, m_nu);
  nu_vec.push_back(nu);
  nu_vec.push_back(nu2);
  return solution_code;
}

// like v9fast, but the parameter space scanning is now factorised out, to allow
// flexibility
int MissingMassCalculatorV2::DitauMassCalculatorV9walk() {

  int nsuccesses = 0;

  int fit_code = 0; // 0==bad, 1==good
  ClearDitauStuff(fDitauStuffFit);
  ClearDitauStuff(fDitauStuffHisto);
  OutputInfo.AveSolRMS = 0.;

  fMfit_all->Reset();
#ifdef SAVELIKELIHOODHISTO
  fMEtP_all->Reset();
  fMEtL_all->Reset();
  fMnu1_all->Reset();
  fMnu2_all->Reset();
  fPhi1_all->Reset();
  fPhi2_all->Reset();
#endif
  fMfit_allNoWeight->Reset();
  fPXfit1->Reset();
  fPYfit1->Reset();
  fPZfit1->Reset();
  fPXfit2->Reset();
  fPYfit2->Reset();
  fPZfit2->Reset();

  // these histograms are used for the floating stopping criterion
  if (fUseFloatStopping) {
    fMmass_split1->Reset();
    fMEtP_split1->Reset();
    fMEtL_split1->Reset();
    fMnu1_split1->Reset();
    fMnu2_split1->Reset();
    fPhi1_split1->Reset();
    fPhi2_split1->Reset();
    fMmass_split2->Reset();
    fMEtP_split2->Reset();
    fMEtL_split2->Reset();
    fMnu1_split2->Reset();
    fMnu2_split2->Reset();
    fPhi1_split2->Reset();
    fPhi2_split2->Reset();
  }

  prob_tmp = 0.0;

  iter1 = 0;

  totalProbSum = 0;
  mtautauSum = 0;

  TVector2 deltamet_vec;

  // initialize a spacewalker, which walks the parameter space according to some
  // algorithm
  SpaceWalkerInit();

  while (SpaceWalkerWalk()) {
    bool paramInsideRange = false;
    m_nsol = 0;

    paramInsideRange = checkAllParamInRange();

    // FIXME if no tau scanning, or symmetric matrices, rotatin is made twice
    // which is inefficient
    const double deltaMetx = m_MEtL * m_metCovPhiCos - m_MEtP * m_metCovPhiSin;
    const double deltaMety = m_MEtL * m_metCovPhiSin + m_MEtP * m_metCovPhiCos;

    //    deltaMetVec.Set(met_smear_x,met_smear_y);
    preparedInput.m_metVec.Set(preparedInput.m_inputMEtX + deltaMetx,
                               preparedInput.m_inputMEtY + deltaMety);

    // save in global variable for speed sake
    preparedInput.m_MEtX = preparedInput.m_metVec.Px();
    preparedInput.m_MEtY = preparedInput.m_metVec.Py();
    preparedInput.m_MEtT = preparedInput.m_metVec.Mod();

    if (paramInsideRange)
      probCalculatorV9fast(m_Phi1, m_Phi2, m_Mnu1, m_Mnu2);

    // DR for markov chain need to enter handleSolution also when zero solutions
    handleSolutions();
    // be careful that with markov, current solution is from now on stored in
    // XYZOldSolVec

    if (m_nsol <= 0)
      continue;

    // for markov, nsuccess more difficult to define. Decide this is the number
    // of independent point accepted (hence without weight)
    nsuccesses = m_markovNAccept;
    m_iterNsuc = m_markovNAccept;

    iter1 += m_nsol;
    fit_code = 1;

  } // while loop

  OutputInfo.NTrials = m_iter0;
  OutputInfo.NSuccesses = nsuccesses;

  if (nsuccesses > 0) {
    OutputInfo.AveSolRMS /= nsuccesses;
  } else {
    OutputInfo.AveSolRMS = -1.;
  }

  double Px1, Py1, Pz1;
  double Px2, Py2, Pz2;
  if (nsuccesses > 0) {

    // note that smoothing can slightly change the integral of the histogram

#ifdef SMOOTH
    fMfit_all->Smooth();
    fMfit_allNoWeight->Smooth();
    fPXfit1->Smooth();
    fPYfit1->Smooth();
    fPZfit1->Smooth();
    fPXfit2->Smooth();
    fPYfit2->Smooth();
    fPZfit2->Smooth();
#endif

    // default max finding method defined in MissingMassCalculator.h
    // note that window defined in terms of number of bin, so depend on binning
    std::vector<double> histInfo(HistInfoV2::MAXHISTINFO);
    fDitauStuffHisto.Mditau_best = maxFromHist(fMfit_all, histInfo);
    double prob_hist = histInfo.at(HistInfoV2::PROB);

    if (prob_hist != 0.0)
      fDitauStuffHisto.Sign_best = -log10(std::abs(prob_hist));
    else {
      // this mean the histogram is empty.
      // possible but very rare if all entries outside histogram range
      // fall back to maximum
      fDitauStuffHisto.Sign_best = -999.;
      fDitauStuffHisto.Mditau_best = fDitauStuffFit.Mditau_best;
    }

    if (fDitauStuffHisto.Mditau_best > 0.0)
      fDitauStuffHisto.RMSoverMPV = fMfit_all->GetRMS() / fDitauStuffHisto.Mditau_best;
    std::vector<double> histInfoOther(HistInfoV2::MAXHISTINFO);
    //---- getting full tau1 momentum
    Px1 = maxFromHist(fPXfit1, histInfoOther);
    Py1 = maxFromHist(fPYfit1, histInfoOther);
    Pz1 = maxFromHist(fPZfit1, histInfoOther);

    //---- getting full tau2 momentum
    Px2 = maxFromHist(fPXfit2, histInfoOther);
    Py2 = maxFromHist(fPYfit2, histInfoOther);
    Pz2 = maxFromHist(fPZfit2, histInfoOther);

    //---- setting 4-vecs
    TLorentzVector fulltau1, fulltau2;

    fulltau1.SetXYZM(Px1, Py1, Pz1, 1.777);
    fulltau2.SetXYZM(Px2, Py2, Pz2, 1.777);
    if (fulltau1.P() < preparedInput.vistau1.P())
      fulltau1 = 1.01 * preparedInput.vistau1; // protection against cases when fitted tau
                                               // momentum is smaller than visible tau momentum
    if (fulltau2.P() < preparedInput.vistau2.P())
      fulltau2 = 1.01 * preparedInput.vistau2; // protection against cases when fitted tau
                                               // momentum is smaller than visible tau momentum
    fDitauStuffHisto.vistau1 = preparedInput.vistau1; // FIXME should also be fitted if tau scan
    fDitauStuffHisto.vistau2 = preparedInput.vistau2;
    fDitauStuffHisto.nutau1 = fulltau1 - preparedInput.vistau1; // these are the original tau vis
    fDitauStuffHisto.nutau2 =
        fulltau2 - preparedInput.vistau2; // FIXME neutrino mass not necessarily zero
  }

  // Note that for v9walk, points outside the METx MEty disk are counted, while
  // this was not the case for v9
  if (preparedInput.fUseVerbose == 1) {
    Info("DiTauMassTools", "Scanning ");
    Info("DiTauMassTools", " Markov ");
    Info("DiTauMassTools", "%s",
         (" V9W niters=" + std::to_string(m_iter0) + " " + std::to_string(iter1)).c_str());
    Info("DiTauMassTools", "%s", (" nFullScan " + std::to_string(m_markovNFullScan)).c_str());
    Info("DiTauMassTools", "%s", (" nRejectNoSol " + std::to_string(m_markovNRejectNoSol)).c_str());
    Info("DiTauMassTools", "%s", (" nRejectMetro " + std::to_string(m_markovNRejectMetropolis)).c_str());
    Info("DiTauMassTools", "%s", (" nAccept " + std::to_string(m_markovNAccept)).c_str());
    Info("DiTauMassTools", "%s",
         (" probsum " + std::to_string(totalProbSum) + " msum " + std::to_string(mtautauSum))
             .c_str());
  }

  if (preparedInput.fUseVerbose == 1) {
    if (fit_code == 0) {
      Info("DiTauMassTools", "%s", ("!!!----> Warning-3 in "
                              "MissingMassCalculator::DitauMassCalculatorV9Walk() : fit status=" +
                              std::to_string(fit_code))
                                 .c_str());
      Info("DiTauMassTools", "%s", "....... No solution is found. Printing input info .......");

      Info("DiTauMassTools", "%s", ("  vis Tau-1: Pt=" + std::to_string(preparedInput.vistau1.Pt()) +
                              "  M=" + std::to_string(preparedInput.vistau1.M()) +
                              " eta=" + std::to_string(preparedInput.vistau1.Eta()) +
                              "  phi=" + std::to_string(preparedInput.vistau1.Phi()) +
                              "  type=" + std::to_string(preparedInput.type_visTau1))
                                 .c_str());
      Info("DiTauMassTools", "%s", ("  vis Tau-2: Pt=" + std::to_string(preparedInput.vistau2.Pt()) +
                              "  M=" + std::to_string(preparedInput.vistau2.M()) +
                              " eta=" + std::to_string(preparedInput.vistau2.Eta()) +
                              "  phi=" + std::to_string(preparedInput.vistau2.Phi()) +
                              "  type=" + std::to_string(preparedInput.type_visTau2))
                                 .c_str());
      Info("DiTauMassTools", "%s", ("  MET=" + std::to_string(preparedInput.MetVec.Mod()) +
                              "  Met_X=" + std::to_string(preparedInput.MetVec.Px()) +
                              "  Met_Y=" + std::to_string(preparedInput.MetVec.Py()))
                                 .c_str());
      Info("DiTauMassTools", " ---------------------------------------------------------- ");
    }
  }

  return fit_code;
}

int MissingMassCalculatorV2::DitauMassCalculatorV9lfv() {

  // debugThisIteration=false;
  debugThisIteration = true;

  int fit_code = 0; // 0==bad, 1==good
  ClearDitauStuff(fDitauStuffFit);
  ClearDitauStuff(fDitauStuffHisto);
  OutputInfo.NTrials = 0;
  OutputInfo.NSuccesses = 0;
  OutputInfo.AveSolRMS = 0.;

  //------- Settings -------------------------------
  int NiterMET = Niter_fit2; // number of iterations for each MET scan loop
  int NiterMnu = Niter_fit3; // number of iterations for Mnu loop
  const double Mtau = 1.777;
  double Mnu_binSize = MnuScanRange / NiterMnu;

  double METresX = preparedInput.METsigmaL; // MET resolution in direction parallel to
                                            // leading jet, for MET scan
  double METresY = preparedInput.METsigmaP; // MET resolution in direction perpendicular to
                                            // leading jet, for MET scan

  //-------- end of Settings
  if (preparedInput.m_tauTypes == TauTypes::ll) { // both tau's are leptonic
    Nsigma_METscan = Nsigma_METscan_ll;
  } else if (preparedInput.m_tauTypes == TauTypes::lh) { // lep had
    Nsigma_METscan = Nsigma_METscan_lh;
  }

  double N_METsigma = Nsigma_METscan; // number of sigmas for MET scan
  double METresX_binSize = 2 * N_METsigma * METresX / NiterMET;
  double METresY_binSize = 2 * N_METsigma * METresY / NiterMET;

  int solution = 0;

  std::vector<TLorentzVector> nu_vec;

  totalProbSum = 0;
  mtautauSum = 0;

  double metprob = 1.0;
  double sign_tmp = 0.0;
  double tauprob = 1.0;
  double totalProb = 0.0;

  prob_tmp = 0.0;

  double met_smear_x = 0.0;
  double met_smear_y = 0.0;
  double met_smearL = 0.0;
  double met_smearP = 0.0;

  double angle1 = 0.0;

  if (fMfit_all) {
    fMfit_all->Reset();
  }
  if (fMfit_allNoWeight) {
    fMfit_allNoWeight->Reset();
  }
  if (fPXfit1) {
    fPXfit1->Reset();
  }
  if (fPYfit1) {
    fPYfit1->Reset();
  }
  if (fPZfit1) {
    fPZfit1->Reset();
  }

  int iter0 = 0;
  iter1 = 0;
  iter2 = 0;
  iter3 = 0;
  iter4 = 0;

  iang1low = 0;
  iang1high = 0;

  int nsuccesses = 0;
  int isol = 0;

  //   double Mvis=(tau_vec1+tau_vec2).M();
  //   TLorentzVector met4vec(0.0,0.0,0.0,0.0);
  //   met4vec.SetPxPyPzE(met_vec.X(),met_vec.Y(),0.0,met_vec.Mod());
  //   double Meff=(tau_vec1+tau_vec2+met4vec).M();
  //   double met_det=met_vec.Mod();

  //---------------------------------------------
  if (preparedInput.m_tauTypes == TauTypes::ll) // dilepton case
  {
    if (preparedInput.fUseVerbose == 1) {
      Info("DiTauMassTools", "Running in dilepton mode");
    }
    double input_metX = preparedInput.MetVec.X();
    double input_metY = preparedInput.MetVec.Y();

    TLorentzVector tau_tmp(0.0, 0.0, 0.0, 0.0);
    TLorentzVector lep_tmp(0.0, 0.0, 0.0, 0.0);
    int tau_type_tmp;
    int tau_ind = 0;

    if (preparedInput.LFVmode == 1) // muon case: H->mu+tau(->ele) decays
    {
      if (preparedInput.vistau1.M() > 0.05 &&
          preparedInput.vistau2.M() < 0.05) // chosing lepton from Higgs decay
      {
        tau_tmp = preparedInput.vistau2;
        lep_tmp = preparedInput.vistau1;
        tau_type_tmp = preparedInput.type_visTau2;
        tau_ind = 2;
      } else {
        tau_tmp = preparedInput.vistau1;
        lep_tmp = preparedInput.vistau2;
        tau_type_tmp = preparedInput.type_visTau1;
        tau_ind = 1;
      }
    }
    if (preparedInput.LFVmode == 0) // electron case: H->ele+tau(->mu) decays
    {
      if (preparedInput.vistau1.M() < 0.05 &&
          preparedInput.vistau2.M() > 0.05) // chosing lepton from Higgs decay
      {
        tau_tmp = preparedInput.vistau2;
        lep_tmp = preparedInput.vistau1;
        tau_type_tmp = preparedInput.type_visTau2;
        tau_ind = 2;
      } else {
        tau_tmp = preparedInput.vistau1;
        lep_tmp = preparedInput.vistau2;
        tau_type_tmp = preparedInput.type_visTau1;
        tau_ind = 1;
      }
    }

    //------- Settings -------------------------------
    double Mlep = tau_tmp.M();
    //       double dMnu_max=MnuScanRange-Mlep;
    //       double Mnu_binSize=dMnu_max/NiterMnu;
    //-------- end of Settings

    //       double M=Mtau;
    double M_nu = 0.0;
    double MnuProb = 1.0;
    //---------------------------------------------
    for (int i3 = 0; i3 < NiterMnu; i3++) //---- loop-3: virtual neutrino mass
    {
      M_nu = Mnu_binSize * i3;
      if (M_nu >= (Mtau - Mlep))
        continue;
      //      M=sqrt(Mtau*Mtau-M_nu*M_nu);
      MnuProb = Prob->MnuProbability(preparedInput, M_nu,
                                     Mnu_binSize); // Mnu probability
      //---------------------------------------------
      for (int i4 = 0; i4 < NiterMET + 1; i4++) // MET_X scan
      {
        met_smearL = METresX_binSize * i4 - N_METsigma * METresX;
        for (int i5 = 0; i5 < NiterMET + 1; i5++) // MET_Y scan
        {
          met_smearP = METresY_binSize * i5 - N_METsigma * METresY;
          if (pow(met_smearL / METresX, 2) + pow(met_smearP / METresY, 2) > pow(N_METsigma, 2))
            continue; // use ellipse instead of square
          met_smear_x = met_smearL * m_metCovPhiCos - met_smearP * m_metCovPhiSin;
          met_smear_y = met_smearL * m_metCovPhiSin + met_smearP * m_metCovPhiCos;
          metvec_tmp.Set(input_metX + met_smear_x, input_metY + met_smear_y);

          solution = NuPsolutionLFV(metvec_tmp, tau_tmp, M_nu, nu_vec);

          ++iter0;

          if (solution < 1)
            continue;
          ++iter1;
          ++isol;
          // if fast sin cos, result to not match exactly nupsolutionv2, so skip
          // test
          // SpeedUp no nested loop to compute individual probability
          int ngoodsol1 = 0;

          metprob = Prob->MetProbability(preparedInput, met_smearL, met_smearP, METresX, METresY);
          if (metprob <= 0)
            continue;
          for (unsigned int j1 = 0; j1 < nu_vec.size(); j1++) {
            if (tau_tmp.E() + nu_vec[j1].E() >= preparedInput.beamEnergy)
              continue;
            const double tau1_tmpp = (tau_tmp + nu_vec[j1]).P();
            angle1 = Angle(nu_vec[j1], tau_tmp);

            if (angle1 < dTheta3DLimit(tau_type_tmp, 0, tau1_tmpp)) {
              ++iang1low;
              continue;
            } // lower 99% bound
            if (angle1 > dTheta3DLimit(tau_type_tmp, 1, tau1_tmpp)) {
              ++iang1high;
              continue;
            } // upper 99% bound
            double tauvecprob1j =
                Prob->dTheta3d_probabilityFast(preparedInput, tau_type_tmp, angle1, tau1_tmpp);
            if (tauvecprob1j == 0.)
              continue;
            tauprob = Prob->TauProbabilityLFV(preparedInput, tau_type_tmp, tau_tmp, nu_vec[j1]);
            totalProb = tauvecprob1j * metprob * MnuProb * tauprob;

            tautau_tmp.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);
            tautau_tmp += tau_tmp;
            tautau_tmp += lep_tmp;
            tautau_tmp += nu_vec[j1];

            const double mtautau = tautau_tmp.M();

            totalProbSum += totalProb;
            mtautauSum += mtautau;

            fit_code = 1; // at least one solution is found

            fMfit_all->Fill(mtautau, totalProb);
            fMfit_allNoWeight->Fill(mtautau, 1.);
            //----------------- using P*fit to fill Px,y,z_tau
            fPXfit1->Fill((tau_tmp + nu_vec[j1]).Px(), totalProb);
            fPYfit1->Fill((tau_tmp + nu_vec[j1]).Py(), totalProb);
            fPZfit1->Fill((tau_tmp + nu_vec[j1]).Pz(), totalProb);

            if (totalProb > prob_tmp) // fill solution with highest probability
            {
              sign_tmp = -log10(totalProb);
              prob_tmp = totalProb;
              fDitauStuffFit.Mditau_best = mtautau;
              fDitauStuffFit.Sign_best = sign_tmp;
              if (tau_ind == 1)
                fDitauStuffFit.nutau1 = nu_vec[j1];
              if (tau_ind == 2)
                fDitauStuffFit.nutau2 = nu_vec[j1];
            }

            ++ngoodsol1;
          }

          if (ngoodsol1 == 0)
            continue;
          iter2 += 1;

          ++nsuccesses;
          iter3 += 1;
        }
      }
    }
  } else if (preparedInput.m_tauTypes == TauTypes::lh) // lepton+tau case
  {
    if (preparedInput.fUseVerbose == 1) {
      Info("DiTauMassTools", "Running in lepton+tau mode");
    }
    //------- Settings -------------------------------

    //----- Stuff below are for Winter 2012 lep-had analysis only; it has to be
    // replaced by a more common scheme once other channels are optimized
    //       TVector2
    //       mht_vec((tau_vec1+tau_vec2).Px(),(tau_vec1+tau_vec2).Py()); //
    //       missing Ht vector for Njet25=0 events const double
    //       mht=mht_vec.Mod();
    double input_metX = preparedInput.MetVec.X();
    double input_metY = preparedInput.MetVec.Y();

    //       double mht_offset=0.0;
    //       if(InputInfo.UseHT)  // use missing Ht (for 0-jet events only for
    //       now)
    //        {
    //          input_metX=-mht_vec.X();
    //          input_metY=-mht_vec.Y();
    //          // dPhi(l-t) dependence of misHt-trueMET
    //          if (m_MMCCalibrationSet==MMCCalibrationSet::MMC2011){
    //            if(met_det<20.0) mht_offset=InputInfo.DelPhiTT>1.8 ?
    //            -161.9+235.5*InputInfo.DelPhiTT-101.6*pow(InputInfo.DelPhiTT,2)+13.57*pow(InputInfo.DelPhiTT,3)
    //            : 12.0; else
    //            mht_offset=115.5-78.1*InputInfo.DelPhiTT+12.83*pow(InputInfo.DelPhiTT,2);

    //          } else if (m_MMCCalibrationSet==MMCCalibrationSet::MMC2012){
    //            if(met_det<20.0)
    //            mht_offset=132.1-79.26*InputInfo.DelPhiTT+11.77*pow(InputInfo.DelPhiTT,2);
    //            else
    //            mht_offset=51.28-23.56*InputInfo.DelPhiTT+2.637*pow(InputInfo.DelPhiTT,2);
    //          }
    //        }
    //       else // use MET (for 0-jet and 1-jet events)
    //        {
    //          input_metX=met_vec.X();
    //          input_metY=met_vec.Y();
    //        }

    TLorentzVector tau_tmp(0.0, 0.0, 0.0, 0.0);
    TLorentzVector lep_tmp(0.0, 0.0, 0.0, 0.0);
    int tau_type_tmp;
    if (preparedInput.type_visTau1 == 8) {
      tau_tmp = preparedInput.vistau2;
      lep_tmp = preparedInput.vistau1;
      tau_type_tmp = preparedInput.type_visTau2;
    }
    if (preparedInput.type_visTau2 == 8) {
      tau_tmp = preparedInput.vistau1;
      lep_tmp = preparedInput.vistau2;
      tau_type_tmp = preparedInput.type_visTau1;
    }

    //---------------------------------------------
    for (int i4 = 0; i4 < NiterMET + 1; i4++) // MET_X scan
    {
      met_smearL = METresX_binSize * i4 - N_METsigma * METresX;
      for (int i5 = 0; i5 < NiterMET + 1; i5++) // MET_Y scan
      {
        met_smearP = METresY_binSize * i5 - N_METsigma * METresY;
        if (pow(met_smearL / METresX, 2) + pow(met_smearP / METresY, 2) > pow(N_METsigma, 2))
          continue; // use ellipse instead of square
        met_smear_x = met_smearL * m_metCovPhiCos - met_smearP * m_metCovPhiSin;
        met_smear_y = met_smearL * m_metCovPhiSin + met_smearP * m_metCovPhiCos;
        metvec_tmp.Set(input_metX + met_smear_x, input_metY + met_smear_y);

        solution = NuPsolutionLFV(metvec_tmp, tau_tmp, 0.0, nu_vec);

        ++iter0;

        if (solution < 1)
          continue;
        ++iter1;
        ++isol;
        // if fast sin cos, result to not match exactly nupsolutionv2, so skip
        // test
        // SpeedUp no nested loop to compute individual probability
        int ngoodsol1 = 0;

        metprob = Prob->MetProbability(preparedInput, met_smearL, met_smearP, METresX, METresY);
        if (metprob <= 0)
          continue;
        for (unsigned int j1 = 0; j1 < nu_vec.size(); j1++) {
          if (tau_tmp.E() + nu_vec[j1].E() >= preparedInput.beamEnergy)
            continue;
          const double tau1_tmpp = (tau_tmp + nu_vec[j1]).P();
          angle1 = Angle(nu_vec[j1], tau_tmp);

          if (angle1 < dTheta3DLimit(tau_type_tmp, 0, tau1_tmpp)) {
            ++iang1low;
            continue;
          } // lower 99% bound
          if (angle1 > dTheta3DLimit(tau_type_tmp, 1, tau1_tmpp)) {
            ++iang1high;
            continue;
          } // upper 99% bound
          double tauvecprob1j =
              Prob->dTheta3d_probabilityFast(preparedInput, tau_type_tmp, angle1, tau1_tmpp);
          if (tauvecprob1j == 0.)
            continue;
          tauprob = Prob->TauProbabilityLFV(preparedInput, tau_type_tmp, tau_tmp, nu_vec[j1]);
          totalProb = tauvecprob1j * metprob * tauprob;

          tautau_tmp.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);
          tautau_tmp += tau_tmp;
          tautau_tmp += lep_tmp;
          tautau_tmp += nu_vec[j1];

          const double mtautau = tautau_tmp.M();

          totalProbSum += totalProb;
          mtautauSum += mtautau;

          fit_code = 1; // at least one solution is found

          fMfit_all->Fill(mtautau, totalProb);
          fMfit_allNoWeight->Fill(mtautau, 1.);
          ////----------------- using P*fit to fill Px,y,z_tau
          // fPXfit1->Fill((tau_tmp+nu_vec[j1]).Px(),totalProb);
          // fPYfit1->Fill((tau_tmp+nu_vec[j1]).Py(),totalProb);
          // fPZfit1->Fill((tau_tmp+nu_vec[j1]).Pz(),totalProb);

          if (totalProb > prob_tmp) // fill solution with highest probability
          {
            sign_tmp = -log10(totalProb);
            prob_tmp = totalProb;
            fDitauStuffFit.Mditau_best = mtautau;
            fDitauStuffFit.Sign_best = sign_tmp;
            if (preparedInput.type_visTau1 == 8) {
              fDitauStuffFit.vistau1 = lep_tmp;
              fDitauStuffFit.vistau2 = tau_tmp;
              fDitauStuffFit.nutau2 = nu_vec[j1];
            } else if (preparedInput.type_visTau2 == 8) {
              fDitauStuffFit.vistau2 = lep_tmp;
              fDitauStuffFit.vistau1 = tau_tmp;
              fDitauStuffFit.nutau1 = nu_vec[j1];
            }
          }

          ++ngoodsol1;
        }

        if (ngoodsol1 == 0)
          continue;
        iter2 += 1;

        ++nsuccesses;
        iter3 += 1;
      }
    }
  } else {
    Info("DiTauMassTools", "Running in an unknown mode?!?!");
  }

  OutputInfo.NTrials = iter0;
  OutputInfo.NSuccesses = iter3;

  if (preparedInput.fUseVerbose == 1) {
    Info("DiTauMassTools", "%s",
         ("SpeedUp niters=" + std::to_string(iter0) + " " + std::to_string(iter1) + " " +
          std::to_string(iter2) + " " + std::to_string(iter3) + "skip:" + std::to_string(iang1low) +
          " " + std::to_string(iang1high))
             .c_str());
  }

  if (fMfit_all->GetEntries() > 0 && iter3 > 0) {
#ifdef SMOOTH
    fMfit_all->Smooth();
    fMfit_allNoWeight->Smooth();
    fPXfit1->Smooth();
    fPYfit1->Smooth();
    fPZfit1->Smooth();
#endif

    // default max finding method defined in MissingMassCalculator.h
    // note that window defined in terms of number of bin, so depend on binning
    std::vector<double> histInfo(HistInfoV2::MAXHISTINFO);
    fDitauStuffHisto.Mditau_best = maxFromHist(fMfit_all, histInfo);
    double prob_hist = histInfo.at(HistInfoV2::PROB);

    if (prob_hist != 0.0)
      fDitauStuffHisto.Sign_best = -log10(std::abs(prob_hist));
    else {
      // this mean the histogram is empty.
      // possible but very rare if all entries outside histogram range
      // fall back to maximum
      fDitauStuffHisto.Sign_best = -999.;
      fDitauStuffHisto.Mditau_best = fDitauStuffFit.Mditau_best;
    }

    if (fDitauStuffHisto.Mditau_best > 0.0)
      fDitauStuffHisto.RMSoverMPV = fMfit_all->GetRMS() / fDitauStuffHisto.Mditau_best;
    std::vector<double> histInfoOther(HistInfoV2::MAXHISTINFO);
    //---- getting Nu1
    double Px1 = maxFromHist(fPXfit1, histInfoOther);
    double Py1 = maxFromHist(fPYfit1, histInfoOther);
    double Pz1 = maxFromHist(fPZfit1, histInfoOther);
    //---- setting 4-vecs
    TLorentzVector nu1_tmp(0.0, 0.0, 0.0, 0.0);
    TLorentzVector nu2_tmp(0.0, 0.0, 0.0, 0.0);
    if (preparedInput.type_visTau1 == 8) {
      nu1_tmp = preparedInput.vistau1;
      nu2_tmp.SetXYZM(Px1, Py1, Pz1, 1.777);
    }
    if (preparedInput.type_visTau2 == 8) {
      nu2_tmp = preparedInput.vistau2;
      nu1_tmp.SetXYZM(Px1, Py1, Pz1, 1.777);
    }
    fDitauStuffHisto.nutau1 = nu1_tmp - preparedInput.vistau1;
    fDitauStuffHisto.nutau2 = nu2_tmp - preparedInput.vistau2;
  }

  if (preparedInput.fUseVerbose == 1) {
    if (fit_code == 0) {
      Info(
          "DiTauMassTools", "%s",
          ("!!!----> Warning-3 in MissingMassCalculator::DitauMassCalculatorV9lfv() : fit status=" +
           std::to_string(fit_code))
              .c_str());
      Info("DiTauMassTools", "....... No solution is found. Printing input info .......");

      Info("DiTauMassTools", "%s", ("  vis Tau-1: Pt="+std::to_string(preparedInput.vistau1.Pt())
                +"  M="+std::to_string(preparedInput.vistau1.M())+" eta="+std::to_string(preparedInput.vistau1.Eta())
                +"  phi="+std::to_string(preparedInput.vistau1.Phi())
                +"  type="+std::to_string(preparedInput.type_visTau1)).c_str());
      Info("DiTauMassTools", "%s", ("  vis Tau-2: Pt="+std::to_string(preparedInput.vistau2.Pt())
                +"  M="+std::to_string(preparedInput.vistau2.M())+" eta="+std::to_string(preparedInput.vistau2.Eta())
                +"  phi="+std::to_string(preparedInput.vistau2.Phi())
                +"  type="+std::to_string(preparedInput.type_visTau2)).c_str());
      Info("DiTauMassTools", "%s", ("  MET="+std::to_string(preparedInput.MetVec.Mod())+"  Met_X="+std::to_string(preparedInput.MetVec.Px())
                +"  Met_Y="+std::to_string(preparedInput.MetVec.Py())).c_str());
      Info("DiTauMassTools", " ---------------------------------------------------------- ");
    }
  }
  return fit_code;
}

// function to fit maximum
Double_t MissingMassCalculatorV2::maxFitting(Double_t *x, Double_t *par)
// Double_t maxFitting(Double_t *x, Double_t *par)
{
  // parabola with parameters max, mean and invwidth
  const double mM = x[0];
  const double mMax = par[0];
  const double mMean = par[1];
  const double mInvWidth2 = par[2]; // if param positif distance between intersection of the
                                    // parabola with x axis: 1/Sqrt(mInvWidth2)
  const double fitval = mMax * (1 - 4 * mInvWidth2 * std::pow(mM - mMean, 2));
  return fitval;
}

// determine the maximum from the histogram
// if input prob not default , compute also some probability
// MaxHistStrategy : different method to find maximum
// TODO should get the array on work on it
// should also find the effective range of the hist

double
MissingMassCalculatorV2::maxFromHist(TH1F *theHist, std::vector<double> &histInfo,
                                              const MaxHistStrategyV2::e maxHistStrategy,
                                              const int winHalfWidth, bool debug) {
  // namespace HistInfo
  // enum e {
  // PROB=0,INTEGRAL,CHI2,DISCRI,TANTHETA,TANTHETAW,FITLENGTH,RMS,RMSVSDISCRI,MAXHISTINFO
  // };
  double maxPos = 0.;
  double prob = 0.;

  for (std::vector<double>::iterator itr = histInfo.begin(); itr != histInfo.end(); ++itr) {
    *itr = -1;
  }

  histInfo[HistInfoV2::INTEGRAL] = theHist->Integral();

  if (maxHistStrategy == MaxHistStrategyV2::MAXBIN ||
      ((maxHistStrategy == MaxHistStrategyV2::MAXBINWINDOW ||
        maxHistStrategy == MaxHistStrategyV2::SLIDINGWINDOW) &&
       winHalfWidth == 0)) {

    // simple max search
    // original version, simple bin maximum
    int max_bin = theHist->GetMaximumBin();
    maxPos = theHist->GetBinCenter(max_bin);

    // FIXME GetEntries is unweighted
    prob = theHist->GetBinContent(max_bin) / double(theHist->GetEntries());
    if (prob > 1.)
      prob = 1.;
    histInfo[HistInfoV2::PROB] = prob;
    return maxPos;
  }

  int hNbins = theHist->GetNbinsX();

  if (maxHistStrategy == MaxHistStrategyV2::MAXBINWINDOW) {
    // average around maximum bin (nearly useless in fact)
    // could be faster
    int max_bin = theHist->GetMaximumBin();
    int iBinMin = max_bin - winHalfWidth;
    if (iBinMin < 0)
      iBinMin = 0;
    int iBinMax = max_bin + winHalfWidth;
    if (iBinMax > hNbins)
      iBinMax = hNbins - 1;
    double sumw = 0;
    double sumx = 0;
    for (int iBin = iBinMin; iBin <= iBinMax; ++iBin) {
      const double weight = theHist->GetBinContent(iBin);
      sumw += weight;
      sumx += weight * theHist->GetBinCenter(iBin);
    }
    maxPos = sumx / sumw;

    // FIXME GetEntries is unweighted
    prob = sumw / theHist->GetEntries();
    if (prob > 1.)
      prob = 1.;

    return maxPos;
  }

  // now compute sliding window anyway
  if (maxHistStrategy != MaxHistStrategyV2::SLIDINGWINDOW &&
      maxHistStrategy != MaxHistStrategyV2::FIT) {
    Error("DiTauMassTools", "%s",
          ("ERROR undefined maxHistStrategy:" + std::to_string(maxHistStrategy)).c_str());
    return -10.;
  }

  // first iteration to find the first and last non zero bin, and the histogram
  // integral (not same as Entries because of weights)
  int lastNonZeroBin = -1;
  int firstNonZeroBin = -1;
  double totalSumw = 0.;
  bool firstNullPart = true;
  for (int iBin = 0; iBin < hNbins; ++iBin) {
    const double weight = theHist->GetBinContent(iBin);
    if (weight > 0) {
      totalSumw += weight;
      lastNonZeroBin = iBin;
      if (firstNullPart) {
        firstNullPart = false;
        firstNonZeroBin = iBin;
      }
    }
  }

  // enlarge first and last non zero bin with window width to avoid side effect
  // (maximum close to the edge)
  firstNonZeroBin = std::max(0, firstNonZeroBin - winHalfWidth - 1);
  lastNonZeroBin = std::min(hNbins - 1, lastNonZeroBin + winHalfWidth + 1);

  // if null histogram quit
  if (firstNullPart)
    return maxPos;

  // determine the size of the sliding window in the fit case

  // sliding window
  const int nwidth = 2 * winHalfWidth + 1;
  double winsum = 0.;

  for (int ibin = 0; ibin < nwidth; ++ibin) {
    winsum += theHist->GetBinContent(ibin);
  }
  double winmax = winsum;

  int max_bin = 0.;
  int iBinL = firstNonZeroBin;
  int iBinR = iBinL + 2 * winHalfWidth;
  bool goingUp = true;

  do {
    ++iBinL;
    ++iBinR;
    const double deltawin = theHist->GetBinContent(iBinR) - theHist->GetBinContent(iBinL - 1);

    if (deltawin < 0) {
      if (goingUp) {
        // if were climbing and now loose more on the left
        // than win on the right. This was a local maxima
        if (winsum > winmax) {
          // global maximum one so far
          winmax = winsum;
          max_bin = (iBinR + iBinL) / 2 - 1;
        }
        goingUp = false; // now going down
      }
    } else {
      // do not care about minima, simply indicate we are going down
      goingUp = true;
    }

    winsum += deltawin;

  } while (iBinR < lastNonZeroBin);

  // now compute average
  int iBinMin = max_bin - winHalfWidth;
  if (iBinMin < 0)
    iBinMin = 0;
  int iBinMax = max_bin + winHalfWidth;
  if (iBinMax >= hNbins)
    iBinMax = hNbins - 1;
  double sumw = 0;
  double sumx = 0;
  for (int iBin = iBinMin; iBin <= iBinMax; ++iBin) {
    const double weight = theHist->GetBinContent(iBin);
    sumw += weight;
    sumx += weight * theHist->GetBinCenter(iBin);
  }

  double maxPosWin = -1.;

  if (sumw > 0.) {
    maxPosWin = sumx / sumw;
  }
  // prob if the fraction of events in the window
  prob = sumw / totalSumw;

  // Definitions of some useful parameters

  const double h_rms = theHist->GetRMS(1);
  histInfo[HistInfoV2::RMS] = h_rms;

  double num = 0;
  double numerator = 0;
  double denominator = 0;
  bool nullBin = false;

  for (int i = iBinMin; i < iBinMax; ++i) {
    double binError = theHist->GetBinError(i);
    if (binError < 1e-10) {
      nullBin = true;
    }
    double binErrorSquare = std::pow(binError, 2);
    num = theHist->GetBinContent(i) / (binErrorSquare);
    numerator = numerator + num;
    denominator = denominator + (1 / (binErrorSquare));
  }
  if (numerator < 1e-10 || denominator < 1e-10 || nullBin == true) {
    histInfo[HistInfoV2::MEANBIN] = -1;
  } else {
    histInfo[HistInfoV2::MEANBIN] = sqrt(1 / denominator) / (numerator / denominator);
  }

  // stop here if only looking for sliding window
  if (maxHistStrategy == MaxHistStrategyV2::SLIDINGWINDOW) {
    return maxPosWin;
  }

  maxPos = maxPosWin;
  // now FIT   maxHistStrategy==MaxHistStrategyV2::FIT

  // now mass fit in range defined by sliding window
  // window will be around maxPos
  const double binWidth = theHist->GetBinCenter(2) - theHist->GetBinCenter(1);
  double fitWidth = (winHalfWidth + 0.5) * binWidth;
  // fit range 2 larger than original window range, 3 if less than 20% of the
  // histogram in slinding window

  if (prob > 0.2) {
    fitWidth *= 2.;
  } else {
    fitWidth *= 3.;
  }
  // fit option : Q == Quiet, no printout S result of the fit returned in
  // TFitResultPtr N do not draw the resulting function

  // if debug plot the fitted function
  TString fitOption = debug ? "QS" : "QNS";
  // root fit
  // Sets initial values
  m_fFitting->SetParameters(sumw / winHalfWidth, maxPos, 0.0025);
  //   TFitResultPtr
  //   fitRes=theHist->Fit("pol2",fitOption,"",maxPos-fitWidth,maxPos+fitWidth);
  TFitResultPtr fitRes =
      theHist->Fit(m_fFitting, fitOption, "", maxPos - fitWidth, maxPos + fitWidth);

  double maxPosFit = -1.;

  if (int(fitRes) == 0) {
    // root fit
    histInfo[HistInfoV2::CHI2] = fitRes->Chi2();
    const double mMax = fitRes->Parameter(0);
    const double mMean = fitRes->Parameter(1);
    const double mInvWidth2 = fitRes->Parameter(2);
    double mMaxError = fitRes->ParError(0);
    m_PrintmMaxError = mMaxError;
    double mMeanError = fitRes->ParError(1);
    m_PrintmMeanError = mMeanError;
    double mInvWidth2Error = fitRes->ParError(2);
    m_PrintmInvWidth2Error = mInvWidth2Error;
    mMeanError = 0.;      // avoid warning
    mInvWidth2Error = 0.; // avoid warning
    const double c = mMax * (1 - 4 * mMean * mMean * mInvWidth2);
    const double b = 8 * mMax * mMean * mInvWidth2;
    const double a = -4 * mMax * mInvWidth2;
    // when built in polynomial fit
    // const double c=fitRes->Parameter(0);
    // const double b=fitRes->Parameter(1);
    // const double a=fitRes->Parameter(2);

    const double h_discri = b * b - 4 * a * c;
    histInfo[HistInfoV2::DISCRI] = h_discri;
    const double sqrth_discri = sqrt(h_discri);
    const double h_fitLength = sqrth_discri / a;
    histInfo[HistInfoV2::FITLENGTH] = h_fitLength;
    histInfo[HistInfoV2::TANTHETA] = 2 * a / sqrth_discri;
    histInfo[HistInfoV2::TANTHETAW] = 2 * a * sumw / sqrth_discri;
    histInfo[HistInfoV2::RMSVSDISCRI] = h_rms / h_fitLength;
    // compute maximum position (only if inverted parabola)
    if (a < 0)
      maxPosFit = -b / (2 * a);
  }

  // keep fit result only if within 80% of fit window, and fit succeeded
  if (maxPosFit >= 0. and std::abs(maxPosFit - maxPosWin) < 0.8 * fitWidth) {
    histInfo[HistInfoV2::PROB] = prob;
    return maxPosFit;
  } else {
    // otherwise keep the weighted average
    // negate prob just to flag such event
    prob = -prob;
    histInfo[HistInfoV2::PROB] = prob;
    return maxPosWin;
  }
}

// compute probability for any input value,can be called from a pure parameter
// scan
// deltametvec is along phijet
// returns number of solution if positive, return code if negative, vector of
// probability and mass
int MissingMassCalculatorV2::probCalculatorV9fast(const double &phi1, const double &phi2,
                                                           const double &M_nu1,
                                                           const double &M_nu2) {
  //  bool debug=true;

  int nsol1;
  int nsol2;

  const int solution = NuPsolutionV3(M_nu1, M_nu2, phi1, phi2, nsol1, nsol2);

  if (solution != 1)
    return -4;
  //  refineSolutions (            M_nu1,M_nu2,
  //                              met_smearL,met_smearP,metvec_tmp.Mod(),
  //                              nsol1, nsol2,m_Mvis,m_Meff);
  refineSolutions(M_nu1, M_nu2, nsol1, nsol2, m_Mvis, m_Meff);

  if (m_nsol <= 0)
    return 0;

  // success

  return m_nsol; // for backward compatibility
}

// nuvecsol1 and nuvecsol2 passed by MMC
int MissingMassCalculatorV2::refineSolutions(const double &M_nu1, const double &M_nu2,
                                                      const int nsol1, const int nsol2,
                                                      const double &Mvis, const double &Meff)

{
  m_nsol = 0;

  if (int(m_probFinalSolVec.size()) < nsolfinalmax)
    Error("DiTauMassTools", "%s",
          ("refineSolutions ERROR probFinalSolVec.size() should be " + std::to_string(nsolfinalmax))
              .c_str());
  if (int(m_mtautauFinalSolVec.size()) < nsolfinalmax)
    Error("DiTauMassTools", "%s",
          ("refineSolutions ERROR mtautauSolVec.size() should be " + std::to_string(nsolfinalmax))
              .c_str());
  if (int(m_nu1FinalSolVec.size()) < nsolfinalmax)
    Error("DiTauMassTools", "%s",
          ("refineSolutions ERROR nu1FinalSolVec.size() should be " + std::to_string(nsolfinalmax))
              .c_str());
  if (int(m_nu2FinalSolVec.size()) < nsolfinalmax)
    Error("DiTauMassTools", "%s",
          ("refineSolutions ERROR nu2FinalSolVec.size() should be " + std::to_string(nsolfinalmax))
              .c_str());
  if (nsol1 > int(nsolmax))
    Error("DiTauMassTools", "%s", ("refineSolutions ERROR nsol1 " + std::to_string(nsol1) +
                             "  > nsolmax !" + std::to_string(nsolmax))
                                .c_str());
  if (nsol2 > int(nsolmax))
    Error("DiTauMassTools", "%s", ("refineSolutions ERROR nsol1 " + std::to_string(nsol2) +
                             "  > nsolmax !" + std::to_string(nsolmax))
                                .c_str());

  int ngoodsol1 = 0;
  int ngoodsol2 = 0;
  double constProb =
      Prob->apply(preparedInput, -99, -99, TLorentzVector(0, 0, 0, 0), TLorentzVector(0, 0, 0, 0),
                  TLorentzVector(0, 0, 0, 0), TLorentzVector(0, 0, 0, 0), true, false, false);

  for (int j1 = 0; j1 < nsol1; ++j1) {
    TLorentzVector &nuvec1_tmpj = nuvecsol1[j1];
    TLorentzVector &tauvecsol1j = tauvecsol1[j1];
    double &tauvecprob1j = tauvecprob1[j1];
    tauvecprob1j = 0.;
    // take first or second solution
    // no time to call rndm, switch more or less randomely, according to an
    // oscillating switch perturbed by m_phi1
    if (nsol1 > 1) {
      if (j1 == 0) { // decide at the first solution which one we will take
        const int pickInt = std::abs(10000 * m_Phi1);
        const int pickDigit = pickInt - 10 * (pickInt / 10);
        if (pickDigit < 5)
          m_switch1 = !m_switch1;
      }
      m_switch1 = !m_switch1;
    }

    if (!m_switch1) {
      nuvec1_tmpj.SetXYZM(nuvec1_tmpj.Px(), nuvec1_tmpj.Py(), nuvec1_tmpj.Pz(), M_nu1);
      tauvecsol1j.SetPxPyPzE(0., 0., 0., 0.);
      tauvecsol1j += nuvec1_tmpj;
      tauvecsol1j += m_tauVec1;
      if (tauvecsol1j.E() >= preparedInput.beamEnergy)
        continue;
      tauvecprob1j = Prob->apply(preparedInput, preparedInput.type_visTau1, -99, m_tauVec1,
                                 TLorentzVector(0, 0, 0, 0), nuvec1_tmpj,
                                 TLorentzVector(0, 0, 0, 0), false, true, false);
      ++ngoodsol1;
    }

    for (int j2 = 0; j2 < nsol2; ++j2) {
      TLorentzVector &nuvec2_tmpj = nuvecsol2[j2];
      TLorentzVector &tauvecsol2j = tauvecsol2[j2];
      double &tauvecprob2j = tauvecprob2[j2];
      if (j1 == 0) {
        tauvecprob2j = 0.;
        // take first or second solution
        // no time to call rndm, switch more or less randomely, according to an
        // oscillating switch perturbed by m_phi2
        if (nsol2 > 1) {
          if (j2 == 0) { // decide at the first solution which one we will take
            const int pickInt = std::abs(10000 * m_Phi2);
            const int pickDigit = pickInt - 10 * int(pickInt / 10);
            if (pickDigit < 5)
              m_switch2 = !m_switch2;
          }
          m_switch2 = !m_switch2;
        }

        if (!m_switch2) {
          nuvec2_tmpj.SetXYZM(nuvec2_tmpj.Px(), nuvec2_tmpj.Py(), nuvec2_tmpj.Pz(), M_nu2);
          tauvecsol2j.SetPxPyPzE(0., 0., 0., 0.);
          tauvecsol2j += nuvec2_tmpj;
          tauvecsol2j += m_tauVec2;
          if (tauvecsol2j.E() >= preparedInput.beamEnergy)
            continue;
          tauvecprob2j = Prob->apply(preparedInput, -99, preparedInput.type_visTau2,
                                     TLorentzVector(0, 0, 0, 0), m_tauVec2,
                                     TLorentzVector(0, 0, 0, 0), nuvec2_tmpj, false, true, false);
          ++ngoodsol2;
        }
      }
      if (tauvecprob1j == 0.)
        continue;
      if (tauvecprob2j == 0.)
        continue;

      double totalProb = 1.;

      tautau_tmp.SetPxPyPzE(0., 0., 0., 0.);
      tautau_tmp += tauvecsol1j;
      tautau_tmp += tauvecsol2j;
      const double mtautau = tautau_tmp.M();

      if (TailCleanUp(m_tauVec1, nuvec1_tmpj, m_tauVec2, nuvec2_tmpj, mtautau, Mvis, Meff,
                      preparedInput.DelPhiTT) == 0) {
        continue;
      }

      totalProb *=
          (constProb * tauvecprob1j * tauvecprob2j *
           Prob->apply(preparedInput, preparedInput.type_visTau1, preparedInput.type_visTau2,
                       m_tauVec1, m_tauVec2, nuvec1_tmpj, nuvec2_tmpj, false, false, true));

      if (totalProb <= 0) {
	      if (preparedInput.fUseVerbose)
		      Warning("DiTauMassTools", "%s",
			      ("null proba solution, rejected "+std::to_string(totalProb)).c_str());
      } else {
        // only count solution with non zero probability
        totalProbSum += totalProb;
        mtautauSum += mtautau;

        if (m_nsol >= int(nsolfinalmax)) {
          Error("DiTauMassTools", "%s",
                ("refineSolutions ERROR nsol getting larger than nsolfinalmax!!! " +
                 std::to_string(m_nsol))
                    .c_str());
          Error("DiTauMassTools", "%s",
                (" j1 " + std::to_string(j1) + " j2 " + std::to_string(j2) + " nsol1 " +
                 std::to_string(nsol1) + " nsol2 " + std::to_string(nsol2))
                    .c_str());
          --m_nsol; // overwrite last solution. However this should really never
                    // happen
        }

        // good solution found, copy in vector
        m_mtautauFinalSolVec[m_nsol] = mtautau;
        m_probFinalSolVec[m_nsol] = totalProb;

        TLorentzVector &nu1Final = m_nu1FinalSolVec[m_nsol];
        TLorentzVector &nu2Final = m_nu2FinalSolVec[m_nsol];
        //      for (int iv=0;iv<4;++iv){

        nu1Final.SetPxPyPzE(nuvec1_tmpj.Px(), nuvec1_tmpj.Py(), nuvec1_tmpj.Pz(), nuvec1_tmpj.E());
        nu2Final.SetPxPyPzE(nuvec2_tmpj.Px(), nuvec2_tmpj.Py(), nuvec2_tmpj.Pz(), nuvec2_tmpj.E());
        // }

        ++m_nsol;
      } // else totalProb<=0

    } // loop j2
  }   // loop j1
  if (ngoodsol1 == 0) {
    return -1;
  }
  if (ngoodsol2 == 0) {
    return -2;
  }
  return m_nsol;
}

int MissingMassCalculatorV2::TailCleanUp(const TLorentzVector &vis1,
                                                  const TLorentzVector &nu1,
                                                  const TLorentzVector &vis2,
                                                  const TLorentzVector &nu2, const double &mmc_mass,
                                                  const double &vis_mass, const double &eff_mass,
                                                  const double &dphiTT) {

  int pass_code = 1;
  if (preparedInput.fUseTailCleanup == 0)
    return pass_code;

  // the Clean-up cuts are specifically for rel16 analyses.
  // the will change in rel17 analyses and after the MMC is updated

  if (preparedInput.m_tauTypes == TauTypes::ll) // lepton-lepton channel
  {
    const double MrecoMvis = mmc_mass / vis_mass;
    if (MrecoMvis > 2.6)
      return 0;
    const double MrecoMeff = mmc_mass / eff_mass;
    if (MrecoMeff > 1.9)
      return 0;
    const double e1p1 = nu1.E() / vis1.P();
    const double e2p2 = nu2.E() / vis2.P();
    if ((e1p1 + e2p2) > 4.5)
      return 0;
    if (e2p2 > 4.0)
      return 0;
    if (e1p1 > 3.0)
      return 0;
  }

  //-------- these are new cuts for lep-had analysis for Moriond
  if (preparedInput.m_tauTypes == TauTypes::lh) // lepton-hadron channel
  {

    if (m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2012 ||
        m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2015 ||
        m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2015HIGHMASS ||
        m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2016MC15C ||
        m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2019 ||
        m_MMCCalibrationSet == MMCCalibrationSetV2::UPGRADE)
      return pass_code; // don't use TailCleanup for 8 & 13 TeV data

    //--------- leave code uncommented to avoid Compilation warnings
    if (Prob->GetUseHT()) {
      const double MrecoMvis = mmc_mass / vis_mass;
      const double MrecoMeff = mmc_mass / eff_mass;
      const double x = dphiTT > 1.5 ? dphiTT : 1.5;
      if ((MrecoMeff + MrecoMvis) > 5.908 - 1.881 * x + 0.2995 * x * x)
        return 0;
    }
  }
  return pass_code;
}

//-------- This function applies correction to compensate for the off-set
double MissingMassCalculatorV2::MassScale(int method, double mass, const int &tau_type1,
                                                   const int &tau_type2) {
  double Fscale = 1.0;
  // calibration for rel16 lep-had analysis only
  if (fApplyMassScale == 1) {
    if (preparedInput.m_tauTypes == TauTypes::lh) {
      if (method != 1)
        return 1.0;
      //      float p0, p1, p2, p3;
      //      if(tau_type1==1 || tau_type2==1) // 1-prong tau's
      //        {
      //          p0=3.014; p1=-71.86; p2=1.018; p3=0.8912;
      //          if(mass>91.2) Fscale=p0/(p1+p2*mass)+p3;
      //          else Fscale=p0/(p1+p2*91.2)+p3;
      //        }
      //      if(tau_type1==3 || tau_type2==3) // 3-prong tau's
      //        {
      //          p0=0.4576; p1=-84.22; p2=0.9783; p3=0.9136;
      //          if(mass>91.2) Fscale=p0/(p1+p2*mass)+p3;
      //          else Fscale=p0/(p1+p2*91.2)+p3;
      //        }
      //      if(Fscale>1.0) Fscale=1.0;
      //      if(Fscale<0.89) Fscale=0.89;

      float p0, p1, p2, p3, p4, p5, p6, p7;
      if ((tau_type1 >= 0 && tau_type1 <= 2) || (tau_type2 >= 0 && tau_type2 <= 2))
        return 1.0;                                                                 // 1-prong tau's
      if ((tau_type1 >= 3 && tau_type1 <= 5) || (tau_type2 >= 3 && tau_type2 <= 5)) // 3-prong tau's
      {
        p0 = 3.014;
        p1 = -71.86;
        p2 = 1.018;
        p3 = 0.8912;
        p4 = 0.4576;
        p5 = -84.22;
        p6 = 0.9783;
        p7 = 0.9136;
        double scale1 = p0 / (p1 + p2 * mass) + p3;
        double scale3 = p4 / (p5 + p6 * mass) + p7;
        if (mass > 91.2)
          Fscale = scale3 / scale1;
        else {
          scale1 = p0 / (p1 + p2 * 91.2) + p3;
          scale3 = p4 / (p5 + p6 * 91.2) + p7;
          Fscale = scale3 / scale1;
        }
      }
      if (Fscale > 1.0)
        Fscale = 1.0;
      if (Fscale < 0.95)
        Fscale = 0.95;
    }
  }
  return 1.0 / Fscale;
}

// note that if MarkovChain the input solutions can be modified
void MissingMassCalculatorV2::handleSolutions()

{

  bool reject = true;
  double totalProbSumSol = 0.;
  double totalProbSumSolOld = 0.;
  bool firstPointWithSol = false;

  for (int isol = 0; isol < m_nsol; ++isol) {
    totalProbSumSol += m_probFinalSolVec[isol];
  }

  double uMC = -1.;
  bool notSureToKeep = true;
  // note : if no solution, the point is treated as having a zero probability
  if (fullParamSpaceScan) {
    reject = false;        // accept anyway in this mode
    notSureToKeep = false; // do not need to test on prob
    if (m_nsol <= 0) {
      // if initial full scaning and no sol : continue
      m_markovNFullScan += 1;
    } else {
      // if we were in in full scan mode and we have a solution, switch it off
      fullParamSpaceScan = false;
      firstPointWithSol = true; // as this is the first point without a solution
                                // there is no old sol
      m_iter0 = 0;              // reset the counter so that separately the full scan pphase
                                // and the markov phase use m_NiterRandomLocal points
      // hack for hh : allow 10 times less iteration for markov than for the
      // fullscan phase
      if (preparedInput.m_tauTypes == TauTypes::hh) {
        m_NiterRandomLocal /= 10;
      }
    }
  }

  if (notSureToKeep) {
    // apply Metropolis algorithm to decide to keep this point.
    // compute the probability of the previous point and the current one
    for (int isol = 0; isol < m_nsolOld; ++isol) {
      totalProbSumSolOld += m_probFinalSolOldVec[isol];
    }

    // accept anyway if null old probability (should only happen for the very
    // first point with a solution)
    if (!firstPointWithSol && totalProbSumSolOld <= 0.) {
      Error("DiTauMassTools", "%s",
            (" ERROR null old probability !!! " + std::to_string(totalProbSumSolOld) + " nsolOld " +
             std::to_string(m_nsolOld))
                .c_str());
      reject = false;
    } else if (totalProbSumSol > totalProbSumSolOld) {
      // if going up, accept anyway
      reject = false;
      // else if (totalProbSumSol < 1E-16) { // if null target probability,
      // reject anyway
    } else if (totalProbSumSol < totalProbSumSolOld * 1E-6) { // if ratio of probability <1e6, point
                                                              // will be accepted only  every 1E6
                                                              // iteration, so can reject anyway
      reject = true;
    } else if (m_nsol <= 0) { // new parametrisation give prob too small to
                              // trigger above condition if no solution is found
      reject = true;
    } else {
      // if going down, reject with a probability
      // 1-totalProbSum/totalProbSumOld)
      uMC = randomGen.Rndm();
      reject = (uMC > totalProbSumSol / totalProbSumSolOld);
    }
  } // if reject

  // proceed with the handling of the solutions wether the old or the new ones

  // optionally fill the vectors with the complete list of points (for all
  // walkstrategy)

  if (reject) {
    // current point reset to the previous one
    // Note : only place where m_MEtP etc... are modified outside spacewalkerXYZ
    m_MEtP = m_MEtP0;
    m_MEtL = m_MEtL0;
    m_Phi1 = m_Phi10;
    m_Phi2 = m_Phi20;
    m_eTau1 = m_eTau10;
    m_eTau2 = m_eTau20;
    if (m_scanMnu1)
      m_Mnu1 = m_Mnu10;
    if (m_scanMnu2)
      m_Mnu2 = m_Mnu20;
  }

  // default case : fill the histogram with solution, using current point
  bool fillSolution = true;
  bool oldToBeUsed = false;

  // now handle the reject or accept cases
  // the tricky thing is that for markov, we accept the old point as soon as a
  // new accepted point is found with a weight equal to one plus the number of
  // rejected point inbetween

  if (reject) {
    fillSolution = false; // do not fill solution, just count number of replication
    m_markovCountDuplicate += 1;
    if (m_nsol <= 0) {
      m_markovNRejectNoSol += 1;
    } else {
      m_markovNRejectMetropolis += 1;
    }

  } else {
    // if accept, will fill solution (except for very first point)  but taking
    // the values from the previous point
    if (!fullParamSpaceScan) {
      m_markovNAccept += 1;
    }
    if (!firstPointWithSol) {
      fillSolution = true;
      oldToBeUsed = true;
    } else {
      fillSolution = false;
    }
  } // else reject

  // if do not fill solution exit now
  // for the first point with solution we need to copy the new sol into the old
  // one before leaving
  if (!fillSolution) {
    if (firstPointWithSol) {
      // current point is the future previous one
      m_nsolOld = m_nsol;
      for (int isol = 0; isol < m_nsol; ++isol) {
        m_probFinalSolOldVec[isol] = m_probFinalSolVec[isol];
        m_mtautauFinalSolOldVec[isol] = m_mtautauFinalSolVec[isol];
        m_nu1FinalSolOldVec[isol] = m_nu1FinalSolVec[isol];
        m_nu2FinalSolOldVec[isol] = m_nu2FinalSolVec[isol];
      }
    }
    return;
  }

  // compute RMS of the different solutions
  double solSum = 0.;
  double solSum2 = 0.;

  for (int isol = 0; isol < m_nsol; ++isol) {
    ++iter5;
    double totalProb;
    double mtautau;
    const TLorentzVector *pnuvec1_tmpj;
    const TLorentzVector *pnuvec2_tmpj;

    if (oldToBeUsed) {
      totalProb = m_probFinalSolOldVec[isol];
      mtautau = m_mtautauFinalSolOldVec[isol];
      pnuvec1_tmpj = &m_nu1FinalSolOldVec[isol];
      pnuvec2_tmpj = &m_nu2FinalSolOldVec[isol];
    } else {
      totalProb = m_probFinalSolVec[isol];
      mtautau = m_mtautauFinalSolVec[isol];
      pnuvec1_tmpj = &m_nu1FinalSolVec[isol];
      pnuvec2_tmpj = &m_nu2FinalSolVec[isol];
    }
    const TLorentzVector &nuvec1_tmpj = *pnuvec1_tmpj;
    const TLorentzVector &nuvec2_tmpj = *pnuvec2_tmpj;

    solSum += mtautau;
    solSum2 += mtautau * mtautau;

    double weight;
    // MarkovChain : accepted events already distributed according to
    // probability distribution, so weight is 1. acutally to have a proper
    // estimate of per bin error, instead of putting several time the same point
    // when metropolis alg reject one (or no solution), rather put it with the
    // multiplicity weight. Should only change the error bars might change if
    // weighted markov chain are used there is also an issue with the 4 very
    // close nearly identical solution
    weight = m_markovCountDuplicate +
             1; // incremented only when a point is rejected, hence need to add 1

    fMfit_all->Fill(mtautau, weight);
#ifdef SAVELIKELIHOODHISTO
    fMEtP_all->Fill(m_MEtP, weight);
    fMEtL_all->Fill(m_MEtL, weight);
    fMnu1_all->Fill(m_Mnu1, weight);
    fMnu2_all->Fill(m_Mnu2, weight);
    fPhi1_all->Fill(m_Phi1, weight);
    fPhi2_all->Fill(m_Phi2, weight);
    if (mtautau != 0. && weight != 0.)
      fMfit_allGraph->SetPoint(m_iter0, mtautau, -TMath::Log(weight));
#endif
    fMfit_allNoWeight->Fill(mtautau, 1.);

    //      fPXfit1->Fill(nuvec1_tmpj.Px(),weight);
    //      fPYfit1->Fill(nuvec1_tmpj.Py(),weight);
    //      fPZfit1->Fill(nuvec1_tmpj.Pz(),weight);
    //      fPXfit2->Fill(nuvec2_tmpj.Px(),weight);
    //      fPYfit2->Fill(nuvec2_tmpj.Py(),weight);
    //      fPZfit2->Fill(nuvec2_tmpj.Pz(),weight);

    //----------------- using P*fit to fill Px,y,z_tau
    // Note that the original vistau are used there deliberately,
    // since they will be subtracted after histogram fitting
    // DR, kudos Antony Lesage : do not create temporary TLV within each Fill,
    // saves 10% CPU
    fPXfit1->Fill(preparedInput.vistau1.Px() + nuvec1_tmpj.Px(), totalProb);
    fPYfit1->Fill(preparedInput.vistau1.Py() + nuvec1_tmpj.Py(), totalProb);
    fPZfit1->Fill(preparedInput.vistau1.Pz() + nuvec1_tmpj.Pz(), totalProb);
    fPXfit2->Fill(preparedInput.vistau2.Px() + nuvec2_tmpj.Px(), totalProb);
    fPYfit2->Fill(preparedInput.vistau2.Py() + nuvec2_tmpj.Py(), totalProb);
    fPZfit2->Fill(preparedInput.vistau2.Pz() + nuvec2_tmpj.Pz(), totalProb);

    // fill histograms for floating stopping criterion, split randomly
    if (fUseFloatStopping) {
      if (randomGen.Rndm() <= 0.5) {
        fMmass_split1->Fill(mtautau, weight);
        fMEtP_split1->Fill(m_MEtP, weight);
        fMEtL_split1->Fill(m_MEtL, weight);
        fMnu1_split1->Fill(m_Mnu1, weight);
        fMnu2_split1->Fill(m_Mnu2, weight);
        fPhi1_split1->Fill(m_Phi1, weight);
        fPhi2_split1->Fill(m_Phi2, weight);
      } else {
        fMmass_split2->Fill(mtautau, weight);
        fMEtP_split2->Fill(m_MEtP, weight);
        fMEtL_split2->Fill(m_MEtL, weight);
        fMnu1_split2->Fill(m_Mnu1, weight);
        fMnu2_split2->Fill(m_Mnu2, weight);
        fPhi1_split2->Fill(m_Phi1, weight);
        fPhi2_split2->Fill(m_Phi2, weight);
      }
    }

    if (totalProb > prob_tmp) // fill solution with highest probability
    {
      prob_tmp = totalProb;
      fDitauStuffFit.Mditau_best = mtautau;
      fDitauStuffFit.Sign_best = -log10(totalProb);
      ;
      fDitauStuffFit.nutau1 = nuvec1_tmpj;
      fDitauStuffFit.nutau2 = nuvec2_tmpj;
      fDitauStuffFit.vistau1 = m_tauVec1;
      fDitauStuffFit.vistau2 = m_tauVec2;
    }
  } // loop on solutions

  m_markovCountDuplicate = 0; // now can reset the duplicate count

  if (oldToBeUsed) {
    // current point is the future previous one
    // TLV copy not super efficient but not dramatic
    m_nsolOld = m_nsol;
    for (int isol = 0; isol < m_nsol; ++isol) {
      m_probFinalSolOldVec[isol] = m_probFinalSolVec[isol];
      m_mtautauFinalSolOldVec[isol] = m_mtautauFinalSolVec[isol];
      m_nu1FinalSolOldVec[isol] = m_nu1FinalSolVec[isol];
      m_nu2FinalSolOldVec[isol] = m_nu2FinalSolVec[isol];
    }
  }

  // compute rms of solutions
  const double solRMS = sqrt(solSum2 / m_nsol - std::pow(solSum / m_nsol, 2));
  OutputInfo.AveSolRMS += solRMS;

  return;
}

void MissingMassCalculatorV2::SpaceWalkerInit() {
  // FIXME could use function pointer to switch between functions
  m_nsolOld = 0;

  double METresX = preparedInput.METsigmaL; // MET resolution in direction parallel to MET
                                            // resolution major axis, for MET scan
  double METresY = preparedInput.METsigmaP; // MET resolution in direction perpendicular to
                                            // to MET resolution major axis, for MET scan

  // precompute some quantities and store in m_ data members
  precomputeCache();
  if (m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2019) {
    if (Prob->GetUseMnuProbability() == true && (preparedInput.m_tauTypes == TauTypes::ll || preparedInput.m_tauTypes == TauTypes::lh) ) Prob->setParamNuMass();
    Prob->setParamAngle(m_tauVec1, 1, preparedInput.type_visTau1);
    Prob->setParamAngle(m_tauVec2, 2, preparedInput.type_visTau2);
    Prob->setParamRatio(1, preparedInput.type_visTau1);
    Prob->setParamRatio(2, preparedInput.type_visTau2);
  }

  if (preparedInput.m_tauTypes == TauTypes::ll) // both tau's are leptonic
  {
    Nsigma_METscan = Nsigma_METscan_ll;
  } else if (preparedInput.m_tauTypes == TauTypes::lh) // lep had
  {
    Nsigma_METscan = Nsigma_METscan_lh;
  } else // hh
  {
    Nsigma_METscan = Nsigma_METscan_hh;
  }

  Nsigma_METscan2 = std::pow(Nsigma_METscan, 2);

  const double deltaPhi1 = MaxDelPhi(preparedInput.type_visTau1, m_tauVec1P, dRmax_tau);
  const double deltaPhi2 = MaxDelPhi(preparedInput.type_visTau2, m_tauVec2P, dRmax_tau);

  m_walkWeight = 1.;

  // dummy initial value to avoid printout with random values
  m_Phi10 = 0.;
  m_Phi20 = 0.;
  m_MEtL0 = 0.;
  m_MEtP0 = 0.;
  m_Mnu10 = 0.;
  m_Mnu20 = 0.;

  m_mTau = 1.777;

  // seeds the random generator in a reproducible way from the phi of both tau;
  double aux = std::abs(m_tauVec1Phi + double(m_tauVec2Phi) / 100. / TMath::Pi()) * 100;
  m_seed = (aux - floor(aux)) * 1E6 * (1 + RndmSeedAltering) + 13;

  randomGen.SetSeed(m_seed);
  // int Niter=Niter_fit1; // number of points for each dR loop
  // int NiterMET=Niter_fit2; // number of iterations for each MET scan loop
  // int NiterMnu=Niter_fit3; // number of iterations for Mnu loop

  // approximately compute the number of points from the grid scanning
  // divide by abritry number to recover timing with still better results
  //    m_NiterRandom=(NiterMET+1)*(NiterMET+1)*4*Niter*Niter/10;

  m_Phi1Min = m_tauVec1Phi - deltaPhi1;
  m_Phi1Max = m_tauVec1Phi + deltaPhi1;
  m_Phi1Range = m_Phi1Max - m_Phi1Min;

  m_Phi2Min = m_tauVec2Phi - deltaPhi2;
  m_Phi2Max = m_tauVec2Phi + deltaPhi2;
  m_Phi2Range = m_Phi2Max - m_Phi2Min;

  m_Mnu1Min = 0.;
  m_scanMnu1 = false;
  m_Mnu1 = m_Mnu1Min;

  // for markov chain use factor 2
  m_NiterRandomLocal = NiterRandom / 2;

  // NiterRandom set by user (default is -1). If negative, defines the default
  // here. no more automatic scaling for ll hl hh
  if (NiterRandom <= 0) {
    m_NiterRandomLocal = 100000; // number of iterations for Markov for lh
    if (preparedInput.m_tauTypes == TauTypes::ll)
      m_NiterRandomLocal *= 2; // multiplied for ll , unchecked
    if (preparedInput.m_tauTypes == TauTypes::hh)
      m_NiterRandomLocal *= 5; // divided for hh ,checked
  } else {
    m_NiterRandomLocal = NiterRandom;
  }

  if (preparedInput.type_visTau1 == 8) {
    //      m_Mnu1Max=m_mTau-m_tauVec1M;
    m_Mnu1Max = MnuScanRange - m_tauVec1M;
    m_Mnu1Range = m_Mnu1Max - m_Mnu1Min;
    m_scanMnu1 = true;
  }

  m_Mnu2Min = 0.;
  m_scanMnu2 = false;
  m_Mnu2 = m_Mnu2Min;
  if (preparedInput.type_visTau2 == 8) {
    //      m_Mnu2Max=m_mTau-m_tauVec2M;
    m_Mnu2Max = MnuScanRange - m_tauVec2M;
    m_Mnu2Range = m_Mnu2Max - m_Mnu2Min;
    m_scanMnu2 = true;
  }

  m_MEtLMin = -Nsigma_METscan * METresX;
  m_MEtLMax = +Nsigma_METscan * METresX;
  m_MEtLRange = m_MEtLMax - m_MEtLMin;

  m_MEtPMin = -Nsigma_METscan * METresY;
  m_MEtPMax = +Nsigma_METscan * METresY;
  m_MEtPRange = m_MEtPMax - m_MEtPMin;

  m_eTau1Min = -1;
  m_eTau1Max = -1;
  m_eTau2Min = -1;
  m_eTau2Max = -1;

  m_switch1 = true;
  m_switch2 = true;

  m_NsucStop = NsucStop;
  m_rmsStop = RMSStop;

  m_iter0 = -1;
  m_iterNuPV3 = 0;
  m_testptn1 = 0;
  m_testptn2 = 0;
  m_testdiscri1 = 0;
  m_testdiscri2 = 0;
  m_nosol1 = 0;
  m_nosol2 = 0;
  m_iterNsuc = 0;
  if (m_meanbinStop > 0) {
    m_meanbinToBeEvaluated = true;
  } else {
    m_meanbinToBeEvaluated = false;
  }

  m_markovCountDuplicate = 0;
  m_markovNRejectNoSol = 0;
  m_markovNRejectMetropolis = 0;
  m_markovNAccept = 0;
  m_markovNFullScan = 0;
  // set full parameter space scannning for the first steps, until a solution is
  // found
  fullParamSpaceScan = true;
  // size of step. Needs to be tune. Start with simple heuristic.
  if (m_ProposalTryMEt < 0) {
    m_MEtProposal = m_MEtPRange / 30.;
  } else {
    m_MEtProposal = m_MEtPRange * m_ProposalTryMEt;
  }
  if (m_ProposalTryPhi < 0) {
    m_PhiProposal = 0.04;
  } else {
    m_PhiProposal = m_ProposalTryPhi;
  }
  // FIXME if m_Mnu1Range !ne m_Mnu2Range same proposal will be done
  if (m_scanMnu1) {
    if (m_ProposalTryMnu < 0) {
      m_MnuProposal = m_Mnu1Range / 10.;
    } else {
      m_MnuProposal = m_Mnu1Range * m_ProposalTryMnu;
    }
  }
  if (m_scanMnu2) {
    if (m_ProposalTryMnu < 0) {
      m_MnuProposal = m_Mnu2Range / 10.;
    } else {
      m_MnuProposal = m_Mnu2Range * m_ProposalTryMnu;
    }
  }
}

// iterator. walk has internal counters, should only be used in a while loop
// so far only implement grid strategy
// act on MMC data member to be fast
bool MissingMassCalculatorV2::SpaceWalkerWalk() {
  preparedInput.m_MEtX = -999.;
  preparedInput.m_MEtY = -999.;

  ++m_iter0;

  if (m_meanbinToBeEvaluated && m_iterNsuc == 500) {
    Info("DiTauMassTools", " in m_meanbinToBeEvaluated && m_iterNsuc==500 ");
    // for markov chain m_iterNsuc is the number of *accepted* points, so there
    // can be several iterations without any increment of m_iterNsuc. Hence need
    // to make sure meanbin is evaluated only once
    m_meanbinToBeEvaluated = false;

    // Meanbin stopping criterion
    std::vector<double> histInfo(HistInfoV2::MAXHISTINFO);
    // SLIDINGWINDOW strategy to avoid doing the parabola fit now given it will
    // not be use
    maxFromHist(fMfit_all, histInfo, MaxHistStrategyV2::SLIDINGWINDOW);
    double meanbin = histInfo.at(HistInfoV2::MEANBIN);
    if (meanbin < 0) {
      m_NsucStop = -1; // no meaningful meanbin switch back to niter criterion
    } else {
      double stopdouble = 500 * std::pow((meanbin / m_meanbinStop), 2);
      int stopint = stopdouble;
      m_NsucStop = stopint;
    }
    if (m_NsucStop < 500)
      return false;
  }
  // should be outside m_meanbinStop test
  if (m_iterNsuc == m_NsucStop)
    return false; // Critere d'arret pour nombre de succes

  if (m_iter0 == m_NiterRandomLocal)
    return false; // for now simple stopping criterion on number of iteration

  // floating stopping criterion, reduces run-time for lh, hh by a factor ~2 and ll by roughly
  // factor ~3 check if every scanned variable and resulting mass thermalised after 10k iterations
  // and then every 1k iterations do this by checking that the means of the split distributions is
  // comparable within 5% of their sigma
  if (m_iter0 >= 10000 && (m_iter0 % 1000) == 0 && fUseFloatStopping) {
    if (fabs(fMEtP_split1->GetMean() - fMEtP_split2->GetMean()) <= 0.05 * fMEtP_split1->GetRMS()) {
      if (fabs(fMEtL_split1->GetMean() - fMEtL_split2->GetMean()) <=
          0.05 * fMEtL_split1->GetRMS()) {
        if (fabs(fMnu1_split1->GetMean() - fMnu1_split2->GetMean()) <=
            0.05 * fMnu1_split1->GetRMS()) {
          if (fabs(fMnu2_split1->GetMean() - fMnu2_split2->GetMean()) <=
              0.05 * fMnu2_split1->GetRMS()) {
            if (fabs(fPhi1_split1->GetMean() - fPhi1_split2->GetMean()) <=
                0.05 * fPhi1_split1->GetRMS()) {
              if (fabs(fPhi2_split1->GetMean() - fPhi2_split2->GetMean()) <=
                  0.05 * fPhi2_split1->GetRMS()) {
                if (fabs(fMmass_split1->GetMean() - fMmass_split2->GetMean()) <=
                    0.05 * fMmass_split1->GetRMS()) {
                  return false;
                }
              }
            }
          }
        }
      }
    }
  }

  if (fullParamSpaceScan) {
    // as long as no solution found need to randomise on the full parameter
    // space

    // cut the corners in MissingET (not optimised at all)
    // not needed if distribution is already gaussian
    do {
      m_MEtP = m_MEtPMin + m_MEtPRange * randomGen.Rndm();
      m_MEtL = m_MEtLMin + m_MEtLRange * randomGen.Rndm();
    } while (!checkMEtInRange());

    if (m_scanMnu1) {
      m_Mnu1 = m_Mnu1Min + m_Mnu1Range * randomGen.Rndm();
    }

    if (m_scanMnu2) {
      m_Mnu2 = m_Mnu2Min + m_Mnu2Range * randomGen.Rndm();
    }

    m_Phi1 = m_Phi1Min + m_Phi1Range * randomGen.Rndm();
    m_Phi2 = m_Phi2Min + m_Phi2Range * randomGen.Rndm();

    return true;
  }

  // here the real markov chain takes place : "propose" the new point
  // note that if one parameter goes outside range, this should not be fixed
  // here but later in handleSolution, otherwise would cause a bias

  //   m_MEtP0 etc... also store the position of the previous Markov Chain step,
  //   which is needed by the algorithm
  m_MEtP0 = m_MEtP;
  m_MEtL0 = m_MEtL;

  m_MEtP = randomGen.Gaus(m_MEtP0, m_MEtProposal);

  m_MEtL = randomGen.Gaus(m_MEtL0, m_MEtProposal);

  if (m_scanMnu1) {
    m_Mnu10 = m_Mnu1;
    m_Mnu1 = randomGen.Gaus(m_Mnu10, m_MnuProposal);
  }

  if (m_scanMnu2) {
    m_Mnu20 = m_Mnu2;
    m_Mnu2 = randomGen.Gaus(m_Mnu20, m_MnuProposal);
  }

  m_Phi10 = m_Phi1;
  m_Phi1 = randomGen.Gaus(m_Phi10, m_PhiProposal);

  m_Phi20 = m_Phi2;

  m_Phi2 = randomGen.Gaus(m_Phi20, m_PhiProposal);

  return true;
}

// compute cached values (this value do not change within one call of MMC,
// except for tau e scanning) return true if cache was already uptodatexs
inline bool MissingMassCalculatorV2::precomputeCache() {

  // copy tau 4 vect. If tau E scanning, these vectors will be modified
  m_tauVec1 = preparedInput.vistau1;
  m_tauVec2 = preparedInput.vistau2;

  const TVector2 &metVec = preparedInput.MetVec;

  bool same = true;
  same = updateDouble(m_tauVec1.Phi(), m_tauVec1Phi) && same;
  same = updateDouble(m_tauVec2.Phi(), m_tauVec2Phi) && same;
  same = updateDouble(m_tauVec1.M(), m_tauVec1M) && same;
  same = updateDouble(m_tauVec2.M(), m_tauVec2M) && same;
  same = updateDouble(m_tauVec1.E(), m_tauVec1E) && same;
  same = updateDouble(m_tauVec2.E(), m_tauVec2E) && same;
  same = updateDouble(m_tauVec1.Px(), m_tauVec1Px) && same;
  same = updateDouble(m_tauVec1.Py(), m_tauVec1Py) && same;
  same = updateDouble(m_tauVec1.Pz(), m_tauVec1Pz) && same;
  same = updateDouble(m_tauVec2.Px(), m_tauVec2Px) && same;
  same = updateDouble(m_tauVec2.Py(), m_tauVec2Py) && same;
  same = updateDouble(m_tauVec2.Pz(), m_tauVec2Pz) && same;
  same = updateDouble(m_tauVec1.P(), m_tauVec1P) && same;
  same = updateDouble(m_tauVec2.P(), m_tauVec2P) && same;

  same = updateDouble(1.777, m_mTau) && same;
  same = updateDouble(std::pow(m_mTau, 2), m_mTau2) && same;
  same = updateDouble(cos(preparedInput.METcovphi), m_metCovPhiCos) && same;
  same = updateDouble(sin(preparedInput.METcovphi), m_metCovPhiSin) && same;
  same = updateDouble((m_tauVec1 + m_tauVec2).M(), m_Mvis) && same;

  TLorentzVector Met4vec;
  Met4vec.SetPxPyPzE(preparedInput.MetVec.X(), preparedInput.MetVec.Y(), 0.0,
                     preparedInput.MetVec.Mod());
  same = updateDouble((m_tauVec1 + m_tauVec2 + Met4vec).M(), m_Meff) && same;

  same = updateDouble(preparedInput.HtOffset, preparedInput.m_HtOffset) && same;
  // note that if useHT met_vec is actually -HT
  same = updateDouble(metVec.X(), preparedInput.m_inputMEtX) && same;
  same = updateDouble(metVec.Y(), preparedInput.m_inputMEtY) && same;
  same = updateDouble(metVec.Mod(), preparedInput.m_inputMEtT) && same;

  return same;
}

// return true if all parameters are within their domain
inline bool MissingMassCalculatorV2::checkAllParamInRange() {

  if (m_scanMnu1) {
    if (m_Mnu1 < m_Mnu1Min)
      return false;
    if (m_Mnu1 > m_Mnu1Max)
      return false;
    if (m_Mnu1 > m_mTau - m_tauVec1M)
      return false;
  }

  if (m_scanMnu2) {
    if (m_Mnu2 < m_Mnu2Min)
      return false;
    if (m_Mnu2 > m_Mnu2Max)
      return false;
    if (m_Mnu2 > m_mTau - m_tauVec2M)
      return false;
  }

  // FIXME note that since there is a coupling between Met and tau, should
  // rigorously test both together however since the 3 sigma range is just a
  // hack, it is probably OK

  if (m_Phi1 < m_Phi1Min)
    return false;
  if (m_Phi1 > m_Phi1Max)
    return false;

  if (m_Phi2 < m_Phi2Min)
    return false;
  if (m_Phi2 > m_Phi2Max)
    return false;

  if (!checkMEtInRange())
    return false;

  return true;
}

// return true if Met is within disk instead of withing square (cut the corners)
inline bool MissingMassCalculatorV2::checkMEtInRange() {
  // check MEt is in allowed range
  // range is 3sigma disk ("cutting the corners")
  if (std::pow(m_MEtL / preparedInput.METsigmaL, 2) +
          std::pow(m_MEtP / preparedInput.METsigmaP, 2) >
      Nsigma_METscan2) {
    return false;
  } else {
    return true;
  }
}

// ----- returns dTheta3D lower and upper boundaries:
// limit_code=0: 99% lower limit
// limit_code=1; 99% upper limit
// limit_code=2; 95% upper limit
double MissingMassCalculatorV2::dTheta3DLimit(const int &tau_type, const int &limit_code,
                                                       const double &P_tau) {

#ifndef WITHDTHETA3DLIM
  // make the test ineffective if desired
  if (limit_code == 0)
    return 0.;
  if (limit_code == 1)
    return 10.;
  if (limit_code == 2)
    return 10.;
#endif

  double limit = 1.0;
  if (limit_code == 0)
    limit = 0.0;
  double par[3] = {0.0, 0.0, 0.0};
  // ---- leptonic tau's
  if (tau_type == 8) {
    if (limit_code == 0) // lower 99% limit
    {
      par[0] = 0.3342;
      par[1] = -0.3376;
      par[2] = -0.001377;
    }
    if (limit_code == 1) // upper 99% limit
    {
      par[0] = 3.243;
      par[1] = -12.87;
      par[2] = 0.009656;
    }
    if (limit_code == 2) // upper 95% limit
    {
      par[0] = 2.927;
      par[1] = -7.911;
      par[2] = 0.007783;
    }
  }
  // ---- 1-prong tau's
  if (tau_type >= 0 && tau_type <= 2) {
    if (limit_code == 0) // lower 99% limit
    {
      par[0] = 0.2673;
      par[1] = -14.8;
      par[2] = -0.0004859;
    }
    if (limit_code == 1) // upper 99% limit
    {
      par[0] = 9.341;
      par[1] = -15.88;
      par[2] = 0.0333;
    }
    if (limit_code == 2) // upper 95% limit
    {
      par[0] = 6.535;
      par[1] = -8.649;
      par[2] = 0.00277;
    }
  }
  // ---- 3-prong tau's
  if (tau_type >= 3 && tau_type <= 5) {
    if (limit_code == 0) // lower 99% limit
    {
      par[0] = 0.2308;
      par[1] = -15.24;
      par[2] = -0.0009458;
    }
    if (limit_code == 1) // upper 99% limit
    {
      par[0] = 14.58;
      par[1] = -6.043;
      par[2] = -0.00928;
    }
    if (limit_code == 2) // upper 95% limit
    {
      par[0] = 8.233;
      par[1] = -0.3018;
      par[2] = -0.009399;
    }
  }

  if (fabs(P_tau + par[1]) > 0.0)
    limit = par[0] / (P_tau + par[1]) + par[2];
  if (limit_code == 0) {
    if (limit < 0.0) {
      limit = 0.0;
    } else if (limit > 0.03) {
      limit = 0.03;
    }
  } else {
    if (limit < 0.0 || limit > 0.5 * TMath::Pi()) {
      limit = 0.5 * TMath::Pi();
    } else if (limit < 0.05 && limit > 0.0) {
      limit = 0.05; // parameterization only runs up to P~220 GeV in this regime
                    // will set an upper bound of 0.05
    }
  }

  return limit;
}

// returns P(nu1) & P(nu2)
int MissingMassCalculatorV2::NuPsolution(TVector2 met_vec, double theta1, double phi1,
                                                  double theta2, double phi2, double &P1,
                                                  double &P2) {
  int solution_code = 0; // 0== no solution, 1==with solution
  P1 = 0.0;
  P2 = 0.0;
  double D = sin(theta1) * sin(theta2) * sin(phi2 - phi1);
  if (fabs(D) > 0.0) // matrix deteriminant is non-zero
  {
    P1 = (met_vec.Px() * sin(phi2) - met_vec.Py() * cos(phi2)) * sin(theta2) / D;
    P2 = (met_vec.Py() * cos(phi1) - met_vec.Px() * sin(phi1)) * sin(theta1) / D;
    if (P1 > 0.0 && P2 > 0.0)
      solution_code = 1;
  }
  return solution_code;
}

// standard collinear approximation
// it returns code=0 if collinear approximation can't be applied
// and code=1 and Mrec if collinear approximation was applied
int MissingMassCalculatorV2::StandardCollApprox(const TLorentzVector &tau_vec1,
                                                         const TLorentzVector &tau_vec2,
                                                         const TVector2 &met_vec, double &Mrec) {
  int code = 0;
  Mrec = 0.0;
  double P_nu1 = 0.0;
  double P_nu2 = 0.0;
  int coll_code = NuPsolution(met_vec, tau_vec1.Theta(), tau_vec1.Phi(), tau_vec2.Theta(),
                              tau_vec2.Phi(), P_nu1, P_nu2);
  if (coll_code == 1) {
    code = 1;
    TLorentzVector nu1(P_nu1 * sin(tau_vec1.Theta()) * cos(tau_vec1.Phi()),
                       P_nu1 * sin(tau_vec1.Theta()) * sin(tau_vec1.Phi()),
                       P_nu1 * cos(tau_vec1.Theta()), P_nu1);
    TLorentzVector nu2(P_nu2 * sin(tau_vec2.Theta()) * cos(tau_vec2.Phi()),
                       P_nu2 * sin(tau_vec2.Theta()) * sin(tau_vec2.Phi()),
                       P_nu2 * cos(tau_vec2.Theta()), P_nu2);
    Mrec = (nu1 + nu2 + tau_vec1 + tau_vec2).M();
  }
  return code;
}

// checks units of input variables, converts into [GeV] if needed, make all
// possible corrections DR new : now a second structure preparedInput is derived
// from the input one which only has direct user input
void MissingMassCalculatorV2::FinalizeSettings(const TLorentzVector tlvTau1,
					       const TLorentzVector tlvTau2,
					       const int mmcType1,
					       const int mmcType2,
					       const float met_x,
					       const float met_y,
					       const float sumet,
					       const int &njets) {
  const double GEV = 1000.;

  // Convert to GeV. In principle, MMC should cope with MeV but should check
  // thoroughly
  TLorentzVector fixedtau1;
  fixedtau1.SetPtEtaPhiM(tlvTau1.Pt(), tlvTau1.Eta(), tlvTau1.Phi(), tlvTau1.M());
  TLorentzVector fixedtau2;
  fixedtau2.SetPtEtaPhiM(tlvTau2.Pt(), tlvTau2.Eta(), tlvTau2.Phi(), tlvTau2.M());

  preparedInput.SetVisTauType(0, mmcType1);
  preparedInput.SetVisTauType(1, mmcType2);
  preparedInput.SetVisTauVec(0, fixedtau1);
  preparedInput.SetVisTauVec(1, fixedtau2);

  if (mmcType1 == 8 && mmcType2 == 8) {
    preparedInput.m_tauTypes = TauTypes::ll;
  } else if (mmcType1 >= 0 && mmcType1 <= 5 && mmcType2 >= 0 && mmcType2 <= 5) {
    preparedInput.m_tauTypes = TauTypes::hh;
  } else {
    preparedInput.m_tauTypes = TauTypes::lh;
  }
  if (preparedInput.fUseVerbose)
    Info("DiTauMassTools", "%s", ("running for tau types "+std::to_string(preparedInput.type_visTau1)+" "+std::to_string(preparedInput.type_visTau2)).c_str());
  TVector2 met_vec(met_x , met_y);
  preparedInput.SetMetVec(met_vec);
  if (preparedInput.fUseVerbose)
    Info("DiTauMassTools", "%s", ("passing SumEt="+std::to_string(sumet)).c_str());
  preparedInput.SetSumEt(sumet);
  preparedInput.SetNjet25(njets);

  // check that the calibration set has been chosen explicitly, otherwise abort
  if (m_MMCCalibrationSet == MMCCalibrationSetV2::MAXMMCCALIBRATIONSET) {
    Error("DiTauMassTools", "MMCCalibrationSet has not been set !. Please use "
                            "fMMC.SetCalibrationSet(MMCCalibrationSetV2::MMC2011)"
                            " or MMC2012. Abort now. ");
    throw; // stop job
  }
  //----------- Re-ordering input info, to make sure there is no dependence of
  // results on input order
  // this might be needed because a random scan is used
  // highest pT tau is always first
  preparedInput.InputReorder = 0; // set flag to 0 by default, i.e. no re-ordering
  if ((preparedInput.type_visTau1 >= 0 && preparedInput.type_visTau1 <= 5) &&
      preparedInput.type_visTau2 == 8) // if hadron-lepton, reorder to have lepton first
  {
    preparedInput.InputReorder =
        1; // re-order to be done, this flag is to be checked in DoOutputInfo()
  } else if (!((preparedInput.type_visTau2 >= 0 && preparedInput.type_visTau2 <= 5) &&
               preparedInput.type_visTau1 == 8)) // if not lep-had nor had lep, reorder if tau1 is
                                                 // after tau2 clockwise
  {
    if (fixPhiRange(preparedInput.vistau1.Phi() - preparedInput.vistau2.Phi()) > 0) {
      preparedInput.InputReorder = 1; // re-order to be done, this flag is to be
                                      // checked in DoOutputInfo()
    }
  }

  if (preparedInput.InputReorder == 1) // copy and re-order
  {
    std::swap(preparedInput.vistau1, preparedInput.vistau2);
    std::swap(preparedInput.type_visTau1, preparedInput.type_visTau2);
    std::swap(preparedInput.Nprong_tau1, preparedInput.Nprong_tau2);
  }
  //--------- re-ordering is done ---------------------------------------

  preparedInput.DelPhiTT =
      fabs(TVector2::Phi_mpi_pi(preparedInput.vistau1.Phi() - preparedInput.vistau2.Phi()));

  for (unsigned int i = 0; i < preparedInput.jet4vecs.size(); i++) {
    // correcting sumEt, give priority to SetMetScanParamsUE()
    if (preparedInput.METScanScheme == 0) {
      if ((preparedInput.METsigmaP < 0.1 || preparedInput.METsigmaL < 0.1) &&
          preparedInput.SumEt > preparedInput.jet4vecs[i].Pt() &&
          preparedInput.jet4vecs[i].Pt() > 20.0) {
        if (preparedInput.fUseVerbose == 1) {
          Info("DiTauMassTools", "correcting sumET");
        }
        preparedInput.SumEt -= preparedInput.jet4vecs[i].Pt();
      }
    }
  }

  // give priority to SetVisTauType, only do this if type_visTau1 and
  // type_visTau2 are not set
  /*if(type_visTau1<0 && type_visTau2<0 && Nprong_tau1>-1 && Nprong_tau2>-1)
    {
      if(Nprong_tau1==0) type_visTau1 = 8; // leptonic tau
      else if( Nprong_tau1==1) type_visTau1 = 0; // set to 1p0n for now, may use
different solution later like explicit integer for this case that pantau info is
not available? else if( Nprong_tau1==3) type_visTau1 = 3; // set to 3p0n for
now, see above if(Nprong_tau2==0) type_visTau2 = 8; // leptonic tau else if(
Nprong_tau2==1) type_visTau2 = 0; // set to 1p0n for now, see above else if(
Nprong_tau2==3) type_visTau2=3; // set to 3p0n for now, see above
    }
  */
  // checking input mass of hadronic tau-1
  // one prong
  //   // checking input mass of hadronic tau-1
  // DRMERGE LFV addition
  if (m_MMCCalibrationSet == MMCCalibrationSetV2::LFVMMC2012) {
    if ((preparedInput.type_visTau1 >= 0 && preparedInput.type_visTau1 <= 2) &&
        preparedInput.vistau1.M() != 1.1) {
      preparedInput.vistau1.SetPtEtaPhiM(preparedInput.vistau1.Pt(), preparedInput.vistau1.Eta(),
                                         preparedInput.vistau1.Phi(), 1.1);
    }
    if ((preparedInput.type_visTau1 >= 3 && preparedInput.type_visTau1 <= 5) &&
        preparedInput.vistau1.M() != 1.35) {
      preparedInput.vistau1.SetPtEtaPhiM(preparedInput.vistau1.Pt(), preparedInput.vistau1.Eta(),
                                         preparedInput.vistau1.Phi(), 1.35);
    }
    // checking input mass of hadronic tau-2
    if ((preparedInput.type_visTau2 >= 0 && preparedInput.type_visTau2 <= 2) &&
        preparedInput.vistau2.M() != 1.1) {
      preparedInput.vistau2.SetPtEtaPhiM(preparedInput.vistau2.Pt(), preparedInput.vistau2.Eta(),
                                         preparedInput.vistau2.Phi(), 1.1);
    }
    if ((preparedInput.type_visTau2 >= 3 && preparedInput.type_visTau2 <= 5) &&
        preparedInput.vistau2.M() != 1.35) {
      preparedInput.vistau2.SetPtEtaPhiM(preparedInput.vistau2.Pt(), preparedInput.vistau2.Eta(),
                                         preparedInput.vistau2.Phi(), 1.35);
    }
  } else {
    // DRMERGE end LFV addition
    if ((preparedInput.type_visTau1 >= 0 && preparedInput.type_visTau1 <= 2) &&
        preparedInput.vistau1.M() != 0.8) {
      preparedInput.vistau1.SetPtEtaPhiM(preparedInput.vistau1.Pt(), preparedInput.vistau1.Eta(),
                                         preparedInput.vistau1.Phi(), 0.8);
    }
    // 3 prong
    if ((preparedInput.type_visTau1 >= 3 && preparedInput.type_visTau1 <= 5) &&
        preparedInput.vistau1.M() != 1.2) {
      preparedInput.vistau1.SetPtEtaPhiM(preparedInput.vistau1.Pt(), preparedInput.vistau1.Eta(),
                                         preparedInput.vistau1.Phi(), 1.2);
    }
    // checking input mass of hadronic tau-2
    // one prong
    if ((preparedInput.type_visTau2 >= 0 && preparedInput.type_visTau2 <= 2) &&
        preparedInput.vistau2.M() != 0.8) {
      preparedInput.vistau2.SetPtEtaPhiM(preparedInput.vistau2.Pt(), preparedInput.vistau2.Eta(),
                                         preparedInput.vistau2.Phi(), 0.8);
    }
    // 3 prong
    if ((preparedInput.type_visTau2 >= 3 && preparedInput.type_visTau2 <= 5) &&
        preparedInput.vistau2.M() != 1.2) {
      preparedInput.vistau2.SetPtEtaPhiM(preparedInput.vistau2.Pt(), preparedInput.vistau2.Eta(),
                                         preparedInput.vistau2.Phi(), 1.2);
    }
  } // DRDRMERGE LFV else closing

  // correcting sumEt for electron pt, give priority to SetMetScanParamsUE()
  // DR20150615 in tag 00-00-11 and before. The following was done before the
  // mass of the hadronic tau was set which mean that sumEt was wrongly
  // corrected for the hadronic tau pt if the hadronic tau mass was set to zero
  // Sasha 08/12/15: don't do electron Pt subtraction for high mass studies; in
  // the future, need to check if lepton Pt needs to be subtracted for both ele
  // and muon
  if (preparedInput.METsigmaP < 0.1 || preparedInput.METsigmaL < 0.1) {
    // T. Davidek: hack for lep-lep -- subtract lepton pT both for muon and
    // electron
    if ((m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2016MC15C ||
         m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2019) &&
        preparedInput.vistau1.M() < 0.12 && preparedInput.vistau2.M() < 0.12) { // lep-lep channel
      if (preparedInput.SumEt > preparedInput.vistau1.Pt())
        preparedInput.SumEt -= preparedInput.vistau1.Pt();
      if (preparedInput.SumEt > preparedInput.vistau2.Pt())
        preparedInput.SumEt -= preparedInput.vistau2.Pt();
    } else {
      // continue with the original code
      if (preparedInput.SumEt > preparedInput.vistau1.Pt() && preparedInput.vistau1.M() < 0.05 &&
          m_MMCCalibrationSet != MMCCalibrationSetV2::MMC2015HIGHMASS) {
        if (preparedInput.fUseVerbose == 1) {
          Info("DiTauMassTools", "Substracting pt1 from sumEt");
        }
        preparedInput.SumEt -= preparedInput.vistau1.Pt();
      }
      if (preparedInput.SumEt > preparedInput.vistau2.Pt() && preparedInput.vistau2.M() < 0.05 &&
          m_MMCCalibrationSet != MMCCalibrationSetV2::MMC2015HIGHMASS) {
        if (preparedInput.fUseVerbose == 1) {
          Info("DiTauMassTools", "Substracting pt2 from sumEt");
        }
        preparedInput.SumEt -= preparedInput.vistau2.Pt();
      }
    }
  }

  // controling TauProbability settings for UPGRADE studies
  if (m_MMCCalibrationSet == MMCCalibrationSetV2::UPGRADE &&
      preparedInput.fUseDefaults == 1) {
    if ((preparedInput.vistau1.M() < 0.12 && preparedInput.vistau2.M() > 0.12) ||
        (preparedInput.vistau2.M() < 0.12 && preparedInput.vistau1.M() > 0.12)) {
      Prob->SetUseTauProbability(true); // lep-had case
    }
    if (preparedInput.vistau1.M() > 0.12 && preparedInput.vistau2.M() > 0.12) {
      Prob->SetUseTauProbability(false); // had-had case
    }
  }

  // change Beam Energy for different running conditions
  if (m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2015 ||
      m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2015HIGHMASS ||
      m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2016MC15C ||
      m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2019 ||
      m_MMCCalibrationSet == MMCCalibrationSetV2::UPGRADE)
    preparedInput.beamEnergy = 6500.0; // 13 TeV running

  //--------------------- pre-set defaults for Run-2. To disable pre-set
  // defaults set fUseDefaults=0
  if (preparedInput.fUseDefaults == 1) {
    if (m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2015HIGHMASS) {
      SetNsigmaMETscan_ll(4.0);
      SetNsigmaMETscan_lh(4.0);
      SetNsigmaMETscan_hh(4.0);
      preparedInput.fUseTailCleanup = 0;
      if ((preparedInput.vistau1.M() < 0.12 && preparedInput.vistau2.M() > 0.12) ||
          (preparedInput.vistau2.M() < 0.12 && preparedInput.vistau1.M() > 0.12))
        Prob->SetUseTauProbability(false); // lep-had
      if (preparedInput.m_tauTypes == TauTypes::hh)
        Prob->SetUseTauProbability(true); // had-had
      Prob->SetUseMnuProbability(false);
    }
  }
  // compute HTOffset if relevant
  if (Prob->GetUseHT()) // use missing Ht for Njet25=0 events
  {
    // dPhi(l-t) dependence of misHt-trueMET
    double HtOffset = 0.;
    // proper for hh
    if (preparedInput.m_tauTypes == TauTypes::hh) {
      // hh
      double x = preparedInput.DelPhiTT;
      HtOffset = 87.5 - 27.0 * x;
    } else {

      // FIXME the condition is really on MET non on HT ?
      if (m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2011) {
        if (preparedInput.MetVec.Mod() < 20.0)
          HtOffset = preparedInput.DelPhiTT > 1.8 ? -161.9 + 235.5 * preparedInput.DelPhiTT -
                                                        101.6 * pow(preparedInput.DelPhiTT, 2) +
                                                        13.57 * pow(preparedInput.DelPhiTT, 3)
                                                  : 12.0;
        else
          HtOffset = 115.5 - 78.1 * preparedInput.DelPhiTT + 12.83 * pow(preparedInput.DelPhiTT, 2);

      } else if (m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2012) {
        if (preparedInput.MetVec.Mod() < 20.0)
          HtOffset = 132.1 - 79.26 * preparedInput.DelPhiTT +
                     11.77 * pow(preparedInput.DelPhiTT,
                                 2); // updated for HCP-2012, 8 TeV
        else
          HtOffset = 51.28 - 23.56 * preparedInput.DelPhiTT +
                     2.637 * pow(preparedInput.DelPhiTT,
                                 2); // updated for HCP-2012, 8 TeV
      } else if (m_MMCCalibrationSet == MMCCalibrationSetV2::MMC2015) {
        if (preparedInput.MetVec.Mod() < 20.0)
          HtOffset = 132.1 - 79.26 * preparedInput.DelPhiTT +
                     11.77 * pow(preparedInput.DelPhiTT,
                                 2); // updated for HCP-2012, 8 TeV
        else
          HtOffset = 51.28 - 23.56 * preparedInput.DelPhiTT +
                     2.637 * pow(preparedInput.DelPhiTT,
                                 2); // updated for HCP-2012, 8 TeV
      }
    }
    preparedInput.HtOffset = HtOffset;

    // if use HT, replace MET with HT
    preparedInput.METsigmaP =
        preparedInput.MHtSigma2; // sigma of 2nd Gaussian for missing Ht resolution
    preparedInput.METsigmaL = preparedInput.MHtSigma2;

    TLorentzVector tauSum = preparedInput.vistau1 + preparedInput.vistau2;
    preparedInput.MetVec.Set(-tauSum.Px(), -tauSum.Py()); // WARNING this replace metvec by -mht
  }
}
