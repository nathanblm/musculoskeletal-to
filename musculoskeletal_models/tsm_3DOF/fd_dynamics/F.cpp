#include <OpenSim/Simulation/Model/Model.h>
#include <OpenSim/Simulation/SimbodyEngine/PinJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/WeldJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/PlanarJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/Joint.h>
#include <OpenSim/Simulation/SimbodyEngine/SpatialTransform.h>
#include <OpenSim/Simulation/SimbodyEngine/CustomJoint.h>
#include <OpenSim/Common/LinearFunction.h>
#include <OpenSim/Common/PolynomialFunction.h>
#include <OpenSim/Common/MultiplierFunction.h>
#include <OpenSim/Common/Constant.h>
#include <OpenSim/Simulation/Model/SmoothSphereHalfSpaceForce.h>
#include <OpenSim/Simulation/SimulationUtilities.h>
#include "SimTKcommon/internal/recorder.h"

#include <iostream>
#include <iterator>
#include <random>
#include <cassert>
#include <algorithm>
#include <vector>
#include <fstream>

using namespace SimTK;
using namespace OpenSim;

constexpr int n_in = 2; 
constexpr int n_out = 1; 
constexpr int nCoordinates = 3; 
constexpr int NX = nCoordinates*2; 
constexpr int NU = nCoordinates; 

template<typename T> 
T value(const Recorder& e) { return e; }; 
template<> 
double value(const Recorder& e) { return e.getValue(); }; 

template<typename T>
int F_generic(const T** arg, T** res) {

	// Definition of model.
	OpenSim::Model* model;
	model = new OpenSim::Model();

	// Definition of bodies.
	OpenSim::Body* thorax;
	thorax = new OpenSim::Body("thorax", 20.86663005555472594210, Vec3(-0.06271883570194151936, -0.15165074549763785106, 0.00000000000000000000), Inertia(1.01423127615892005871, 0.43156618119044198867, 1.01423127615892005871, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addBody(thorax);

	OpenSim::Body* clavicle;
	clavicle = new OpenSim::Body("clavicle", 0.23027421524576702216, Vec3(-0.01438354243310052856, 0.00595746901394037486, 0.06136172479900511845), Inertia(0.00023667930695988186, 0.00023667930695988186, 0.00004333790859916070, -0.00001851765077880033, -0.00006823638859814462, 0.00005239874879497059));
	model->addBody(clavicle);

	OpenSim::Body* scapula;
	scapula = new OpenSim::Body("scapula", 0.35591058418327081725, Vec3(-0.04767210943305850296, -0.03376314594331233931, -0.04175532303324564626), Inertia(0.00078311608990092778, 0.00072446748308865937, 0.00078311608990092778, 0.00028312352957027363, 0.00025778440319030612, 0.00015190073029839582));
	model->addBody(scapula);

	OpenSim::Body* humerus;
	humerus = new OpenSim::Body("humerus", 1.76501842582918255786, Vec3(0.02195011773694669627, -0.11878719323199136981, -0.01313620600029631776), Inertia(0.01021050243634119559, 0.00212294836026442326, 0.01021050243634119559, -0.00028909835589940511, -0.00019348702642473181, 0.00010227353907687952));
	model->addBody(humerus);

	OpenSim::Body* ulna;
	ulna = new OpenSim::Body("ulna", 0.86632238509629322376, Vec3(0.00971799999999999921, -0.09594999999999999363, 0.02428999999999999909), Inertia(0.00424265183255788901, 0.00090370913780514440, 0.00387427083102413663, 0.00024838284975754575, -0.00005968556014211773, 0.00085589798654225316));
	model->addBody(ulna);

	OpenSim::Body* radius;
	radius = new OpenSim::Body("radius", 0.18309319565592518941, Vec3(0.03363000000000000017, -0.18155999999999999917, 0.01559999999999999928), Inertia(0.00034377001547383903, 0.00006943589984228773, 0.00003155535983350833, 0.00002362341146005816, -0.00000332326690745344, 0.00005030360144348150));
	model->addBody(radius);

	OpenSim::Body* hand;
	hand = new OpenSim::Body("hand", 0.41148941660685245392, Vec3(0.00059999999999999995, -0.09049999999999999711, -0.03649999999999999772), Inertia(0.00050060626740342222, 0.00014923349508941846, 0.00050060626740342222, 0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addBody(hand);

	// Definition of joints.
	OpenSim::WeldJoint* ground_thorax;
	ground_thorax = new OpenSim::WeldJoint("ground_thorax", model->getGround(), Vec3(0.00000000000000000000, 0.80000000000000004441, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *thorax, Vec3(0.00493743204915335961, -0.00312319621307351018, 0.00079313938267933396), Vec3(-0.08009891203544949878, 0.04933249610946970065, 0.18634788463002599501));
	model->addJoint(ground_thorax);

	OpenSim::WeldJoint* sternoclavicular;
	sternoclavicular = new OpenSim::WeldJoint("sternoclavicular", *thorax, Vec3(0.00671228646570200347, 0.00707226869066925230, 0.02712669885914475304), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *clavicle, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-0.05087815549008418980, 0.29661484728124087873, -0.05264740847349048958));
	model->addJoint(sternoclavicular);

	OpenSim::WeldJoint* scapulothoracic;
	scapulothoracic = new OpenSim::WeldJoint("scapulothoracic", *thorax, Vec3(-0.03122459530455306762, -0.01765513824698549880, 0.07487367001850048220), Vec3(0.00000000000000000000, -1.07000000000000006217, 0.00000000000000000000), *scapula, Vec3(0.03061478000000000116, -0.04410660999999999748, -0.06255534999999999568), Vec3(-0.45703948999999999270, -1.10563225999999992233, -0.44420994000000002533));
	model->addJoint(scapulothoracic);

	SpatialTransform st_GlenoHumeral;
	st_GlenoHumeral[0].setCoordinateNames(OpenSim::Array<std::string>("plane_elv", 1, 1));
	st_GlenoHumeral[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_GlenoHumeral[0].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_GlenoHumeral[1].setCoordinateNames(OpenSim::Array<std::string>("shoulder_elv", 1, 1));
	st_GlenoHumeral[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_GlenoHumeral[1].setAxis(Vec3(-1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_GlenoHumeral[2].setFunction(new Constant(1.02883164408187566252));
	st_GlenoHumeral[2].setAxis(Vec3(-0.08459999999999999465, 0.99470000000000002860, -0.05840000000000000052));
	st_GlenoHumeral[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 0.87167774563804756305));
	st_GlenoHumeral[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_GlenoHumeral[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 0.96383680480477873509));
	st_GlenoHumeral[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_GlenoHumeral[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 0.95484232176803507386));
	st_GlenoHumeral[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* GlenoHumeral;
	GlenoHumeral = new OpenSim::CustomJoint("GlenoHumeral", *scapula, Vec3(-0.00832452451869142816, -0.03277044672940921527, 0.00859358429549605951), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *humerus, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_GlenoHumeral);
	model->addJoint(GlenoHumeral);

	SpatialTransform st_elbow;
	st_elbow[0].setFunction(new Constant(0.00000000000000000000));
	st_elbow[0].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_elbow[1].setFunction(new Constant(0.00000000000000000000));
	st_elbow[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_elbow[2].setCoordinateNames(OpenSim::Array<std::string>("elbow_flexion", 1, 1));
	st_elbow[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_elbow[2].setAxis(Vec3(0.04940001000000000114, 0.03660001000000000226, 0.99810825000000003016));
	st_elbow[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.21540034633552695986));
	st_elbow[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_elbow[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 0.84787653027152931617));
	st_elbow[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_elbow[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.03029392688830179203));
	st_elbow[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* elbow;
	elbow = new OpenSim::CustomJoint("elbow", *humerus, Vec3(0.00741394544881170906, -0.24622301325984491394, -0.01267257156393734006), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *ulna, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_elbow);
	model->addJoint(elbow);

	OpenSim::WeldJoint* radioulnar;
	radioulnar = new OpenSim::WeldJoint("radioulnar", *ulna, Vec3(0.00040000000000000002, -0.01150299999999999934, 0.01999899999999999942), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *radius, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.03662896362136616735, -0.59356722536442585447, 0.08507301783456396038));
	model->addJoint(radioulnar);

	OpenSim::WeldJoint* rc;
	rc = new OpenSim::WeldJoint("rc", *radius, Vec3(0.01799999999999999864, -0.24199999999999999289, 0.02500000000000000139), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *hand, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addJoint(rc);

	// Definition of contacts.
	// Initialize system.
	SimTK::State* state;
	state = new State(model->initSystem());

	// Read inputs.
	std::vector<T> x(arg[0], arg[0] + NX);
	std::vector<T> u(arg[1], arg[1] + NU);

	// States and controls.
	T ua[NU];
	Vector QsUs(NX);
	/// States
	for (int i = 0; i < NX; ++i) QsUs[i] = x[i];
	/// Controls
	/// OpenSim and Simbody have different state orders.
	auto indicesOSInSimbody = getIndicesOpenSimInSimbody(*model);
	for (int i = 0; i < NU; ++i) ua[i] = u[indicesOSInSimbody[i]];

	// Set state variables and realize.
	model->setStateVariableValues(*state, QsUs);
	model->realizeDynamics(*state);

	// Compute residual forces.
	/// Set appliedMobilityForces (# mobilities). They are the controls of our problem
	Vector appliedMobilityForces(nCoordinates);
	appliedMobilityForces.setToZero();
	for (int i = 0; i < NU; ++i) appliedMobilityForces[i] = ua[i];
	/// Set appliedBodyForces (# bodies + ground).
	Vector_<SpatialVec> appliedBodyForces;
	int nbodies = model->getBodySet().getSize() + 1;
	appliedBodyForces.resize(nbodies);
	appliedBodyForces.setToZero();
	/// Set gravity.
	Vec3 gravity(0);
	gravity[1] = -9.80664999999999942304;
	/// Add weights to appliedBodyForces.
	for (int i = 0; i < model->getBodySet().getSize(); ++i) {
		model->getMatterSubsystem().addInStationForce(*state,
		model->getBodySet().get(i).getMobilizedBodyIndex(),
		model->getBodySet().get(i).getMassCenter(),
		model->getBodySet().get(i).getMass()*gravity, appliedBodyForces);
	}
	/// Add contact forces to appliedBodyForces.
	/// Calculate induced accelerations.
	Vector_<SpatialVec> A_GB;
	A_GB.resize(nbodies);
	A_GB.setToZero();;
	Vector udot(nCoordinates);
	udot.setToZero();
	model->getMatterSubsystem().calcAccelerationIgnoringConstraints(*state,
			appliedMobilityForces, appliedBodyForces, udot, A_GB);

	/// Outputs.
	/// Coordinate speeds (taken directly from inputs)
	for (int i = 0; i < NU; ++i) res[0][2*i] =
			value<T>(QsUs[2*i+1]);

	/// Induced accelerations (OpenSim and Simbody have different state orders).
	auto indicesSimbodyInOS = getIndicesSimbodyInOpenSim(*model);
	for (int i = 0; i < NU; ++i) res[0][2*i+1] =
			value<T>(udot[indicesSimbodyInOS[i]]);

	return 0;
}

int main() {
	Recorder x[NX];
	Recorder u[NU];
	Recorder x_dot[NX];
	for (int i = 0; i < NX; ++i) x[i] <<= 0;
	for (int i = 0; i < NU; ++i) u[i] <<= 0;
	const Recorder* Recorder_arg[n_in] = { x,u };
	Recorder* Recorder_res[n_out] = { x_dot };
	F_generic<Recorder>(Recorder_arg, Recorder_res);
	double res[NX];
	const Recorder symbolic_zero = x[0] - x[0];
	for (int i = 0; i < NX; ++i) (Recorder_res[0][i] + symbolic_zero) >>= res[i];
	Recorder::stop_recording();
	return 0;
}
