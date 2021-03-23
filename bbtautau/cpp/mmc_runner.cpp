#include <TH1D.h>
#include <TFile.h>
#include <TRandom3.h>
#include <TVector2.h>
#include <TLorentzVector.h>
#include <TEfficiency.h>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <vector>
#include <ostream>
#include <sstream>
#include <iostream>
#include <map>
#include "standaloneMMC/MissingMassCalculatorV2.h"

class histPtDict
{
public:
  static DiTauMassTools::MissingMassCalculatorV2 m_mmc;
  static DiTauMassTools::MissingMassCalculatorV2 m_mmc_ll;
};


DiTauMassTools::MissingMassCalculatorV2 createMMC(bool is_leplep = false)
{
  // Load in MMC tool
  DiTauMassTools::MissingMassCalculatorV2 myMMC(DiTauMassTools::MMCCalibrationSetV2::MMC2019, "MMC_params_v1_fixed.root");

  // Load configuration - note, that this one only applies to lh and hh, we will need a different handle for ll
  myMMC.SetUseFloatStopping(true);
  myMMC.Prob->SetUseTauProbability(1);

  if(is_leplep)
    {
      // Dedicated leplep configuration
      myMMC.SetNsigmaMETscan(4.0);
      myMMC.preparedInput.SetUseTailCleanup(0);
      myMMC.preparedInput.SetUseVerbose(0);
      myMMC.SetNiterFit2(30);
      myMMC.SetNiterFit3(10);
      myMMC.Prob->SetUseDphiLL(1);

      // leplep custom config
      //mmc_njet_min_pt = 30. <-> same as default
      
      //mmc_conf_n_sigma_met = 4.0
      //mmc_conf_use_tail_cleanup = 0 
      //mmc_conf_use_verbose = 0 
      //mmc_conf_n_iter_fit2 = 30
      //mmc_conf_n_iter_fit3 = 10
      //mmc_conf_use_defaults = -1
      //mmc_conf_use_efficiency_recovery = -1
      //mmc_conf_use_METDPhiLL = 1
    }

  return myMMC;

}

DiTauMassTools::MissingMassCalculatorV2 histPtDict::m_mmc = createMMC();
DiTauMassTools::MissingMassCalculatorV2 histPtDict::m_mmc_ll = createMMC(true);


namespace HackedMMC {
  
  enum RecoParticleType {
    none, // = 0
    muon, // 1,
    electron, //2
    tau, // = 3
  };

  enum Channel
    {
      hh, // = 0
      eh, // = 1
      muh, // = 2
      ee, // = 3
      mumu, // = 4
      emu , // = 5
    };


  float x2_corr(float met_et, float met_phi, 
		float tau_0_pt, float tau_0_eta, float tau_0_phi,
		float tau_1_pt, float tau_1_eta, float tau_1_phi,
		int channel=HackedMMC::hh)
  {
    

    // Calculate x2
    float numerator = met_et*sin(met_phi) - met_et*cos(met_phi) * tan(tau_0_phi);
    float denominator = sin(tau_1_phi) - cos(tau_1_phi) * tan(tau_0_phi);

    // Sub-leading neutrino pT
    float col_lep2_pt = numerator / denominator;
    
    return tau_1_pt / (tau_1_pt+col_lep2_pt);
  }

  float x1_corr(float met_et, float met_phi, 
                float tau_0_pt, float tau_0_eta, float tau_0_phi,
		float tau_1_pt, float tau_1_eta, float tau_1_phi,
                int channel=HackedMMC::hh)
  {

    // Calculate x1
    float numerator = met_et*sin(met_phi) - met_et*cos(met_phi) * tan(tau_0_phi);
    float denominator = sin(tau_1_phi) - cos(tau_1_phi) * tan(tau_0_phi);

    // Sub-leading neutrino pT
    float col_lep2_pt = numerator / denominator;

    // Convert to leading neutrino pT
    float col_lep1_pt = met_et*cos(met_phi) - col_lep2_pt*cos(tau_1_phi)/cos(tau_0_phi);
			 
    return tau_0_pt / (tau_0_pt+col_lep1_pt);
  }




  // float m_coll_corr(float met_et, float met_phi,
  // 		    float tau_0_pt, float tau_0_eta, float tau_0_phi,
  // 		    float tau_1_pt, float tau_1_eta, float tau_1_phi,
  // 		    int event_number, int channel=PtCorrHelper::hh)
  // {
  //   // Check if buffered values are correct
  //   if(event_number!=histPtDict::m_currentEvent)
  //     correct_oversample(tau_0_pt, tau_0_eta, tau_0_phi, tau_1_pt, tau_1_eta, tau_1_phi, event_number, channel);
    
  //   float x1 = x1_corr(met_et, met_phi, tau_0_pt, tau_0_eta, tau_0_phi, tau_1_pt, tau_1_eta, tau_1_phi, event_number, channel);
  //   float x2 = x2_corr(met_et, met_phi, tau_0_pt, tau_0_eta, tau_0_phi, tau_1_pt, tau_1_eta, tau_1_phi, event_number, channel);
  //   float m_vis = ditau_m_corr(tau_0_pt, tau_0_eta, tau_0_phi, tau_1_pt, tau_1_eta, tau_1_phi, event_number, channel);
    
  //   if( x1 * x2 != 0)
  //     return m_vis / std::sqrt(x1*x2);
  //   else
  //     return -1;
  // }


  float MMC(float met_et, float met_phi, float met_sumet, int njets,
	    float tau_0_pt, float tau_0_eta, float tau_0_phi,
	    float tau_1_pt, float tau_1_eta, float tau_1_phi,
	    int event_number,
	    int channel=HackedMMC::hh)
  {
    TVector2 met;
    met.SetMagPhi(met_et, met_phi);


    int tau0=8;
    int tau1=8;

    TRandom3 random_object_picker(event_number+1234567);
    float rnd_number = 0;

    // pantau convention: 0:1p0n, 1:1p1n, 2:1pXn, 3:3p0n, 4:3pXn, 5:2p/4p
    // use truth decay rates for now, this will need to be replaced
    // if(histPtDict::m_taus_type[0]==PtCorrHelper::tau)
    //   {
    // 	rnd_number = random_object_picker.Rndm();
    // 	if(rnd_number < 1.0)
    // 	  tau0=4;
    // 	if(rnd_number < 0.923)
    // 	  tau0=3;
    // 	if(rnd_number < 0.763)
    // 	  tau0=2;
    // 	if(rnd_number < 0.61)
    // 	  tau0=1;
    // 	if(rnd_number < 0.19)
    // 	  tau0=0;
    //   }
    // if(histPtDict::m_taus_type[1]==PtCorrHelper::tau)
    //   {
    // 	rnd_number = random_object_picker.Rndm();
    // 	if(rnd_number < 1.0)
    // 	  tau1=4;
    // 	if(rnd_number < 0.923)
    // 	  tau1=3;
    // 	if(rnd_number < 0.763)
    // 	  tau1=2;
    // 	if(rnd_number < 0.61)
    // 	  tau1=1;
    // 	if(rnd_number < 0.19)
    // 	  tau1=0;	
    //   }

    bool leplep = false;
    if (channel == HackedMMC::emu)
      leplep = true;
    if (channel == HackedMMC::ee)
      leplep = true;
    if (channel == HackedMMC::mumu)
      leplep = true;
    
      
    if(!leplep)
      {
	histPtDict::m_mmc.MMCforBoom(tau_0_pt, tau_0_eta, tau_0_phi, 0,
				     tau_1_pt, tau_1_eta, tau_1_phi, 0,
				     tau0, tau1, met.Px(), met.Py(), met_sumet, njets, event_number);
	
	if(histPtDict::m_mmc.OutputInfo.GetFitStatus() <= 0)
	  return -1;
	else
	  return histPtDict::m_mmc.OutputInfo.GetFittedMass(DiTauMassTools::MMCFitMethodV2::MLM);
      } else 
      {
	histPtDict::m_mmc_ll.MMCforBoom(tau_0_pt, tau_0_eta, tau_0_phi, 0,
					tau_1_pt, tau_1_eta, tau_1_phi, 0,
					tau0, tau1, met.Px(), met.Py(), met_sumet, njets, event_number);
	
	if(histPtDict::m_mmc_ll.OutputInfo.GetFitStatus() <= 0)
	  return -1;
	else
	  return histPtDict::m_mmc_ll.OutputInfo.GetFittedMass(DiTauMassTools::MMCFitMethodV2::MLM);
      }
  }
}
