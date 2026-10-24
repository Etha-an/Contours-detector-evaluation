#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/features2d.hpp>

#include "laplacianZC.h"
#include "edgeDetector.h"

std::vector<std::string> noms = {
	"basket", "bear", "bear_2", "bear_3", "bear_4", "bear_5", "bear_6", "bear_7",
	"bear_8", "bears", "brush", "buffalo", "buffalo_2", "elephant", "elephant_2", "elephant_3",
	"elephant_4", "elephants", "elephants_2", "gazelle", "gazelle_2", "gnu", "gnu_2", "goat",
	"goat_2", "goat_3", "goat_4", "goat_5", "goat_6", "goat_7", "goat_8", "golfcart",
	"hyena", "lions", "rino", "rino_2", "rino_3", "tiger", "tire", "turtle"
};

std::string nomsDetecteurs[3] = { "Sobel", "Laplace", "Canny" };

void ajouterTrackbar(const std::string& nom, int valeur, int max) {
	cv::createTrackbar(nom, "Reglages", nullptr, max);
	cv::setTrackbarPos(nom, "Reglages", valeur);
}

int lire(const std::string& nom) {
	return cv::getTrackbarPos(nom, "Reglages");
}

int compterPixels(const cv::Mat& image) {
	int n = 0;
	for (int y = 0; y < image.rows; y++) {
		const uchar* ligne = image.ptr<uchar>(y);
		for (int x = 0; x < image.cols; x++) {
			if (ligne[x]) n++;
		}
	}
	return n;
}

cv::Mat detecterContours(const cv::Mat& image, int detecteur) {

	cv::Mat lisse = image.clone();
	int flou = lire("Flou");
	if (flou > 0)
		cv::GaussianBlur(image, lisse, cv::Size(), flou / 10.0);

	int ouverture = 2 * lire("Ouverture") + 3;
	cv::Mat contours;

	if (detecteur == 0) {
		EdgeDetector ed;
		ed.setAperture(ouverture);
		ed.computeSobel(lisse);
		cv::threshold(ed.getSobelImage(), contours, lire("Sobel"), 255, cv::THRESH_BINARY);
	}
	else if (detecteur == 1) {
		LaplacianZC laplacian;
		laplacian.setAperture(ouverture);
		cv::Mat laplace = laplacian.computeLaplacian(lisse);
		cv::Mat passages = laplacian.getZeroCrossings(laplace);

		cv::Mat laplaceImage = laplacian.getLaplacianImage();
		int seuil = lire("Laplace");
		cv::Mat forts = (laplaceImage > 128 + seuil) | (laplaceImage < 128 - seuil);

		contours = passages & forts;
	}
	else {
		cv::Canny(lisse, contours, lire("Canny bas"), lire("Canny haut"));
	}

	return contours;
}

void evaluer(const cv::Mat& contours, const cv::Mat& reference, bool voisinage,
	int& detectes, int& contoursReference, int& corrects, int& fauxPositifs, int& fauxNegatifs,
	cv::Mat& comparaison) {

	cv::Mat zoneReference = reference.clone();
	cv::Mat zoneDetectee = contours.clone();
	if (voisinage) {
		cv::dilate(reference, zoneReference, cv::Mat());
		cv::dilate(contours, zoneDetectee, cv::Mat());
	}

	cv::Mat bons = contours & zoneReference;
	cv::Mat faux = contours - bons;
	cv::Mat manques = reference - (reference & zoneDetectee);

	detectes = compterPixels(contours);
	contoursReference = compterPixels(reference);
	corrects = compterPixels(bons);
	fauxPositifs = detectes - corrects;
	fauxNegatifs = compterPixels(manques);

	std::vector<cv::Mat> plans;
	plans.push_back(manques);
	plans.push_back(bons);
	plans.push_back(faux);
	cv::merge(plans, comparaison);
}

void traiterLot(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& references,
	int detecteur, bool details) {

	bool voisinage = lire("Voisinage") == 1;
	double sommeP = 0, sommeTFP = 0, sommeTFN = 0;

	std::cout << std::fixed << std::setprecision(3);

	if (details) {
		std::cout << std::endl << "=== " << nomsDetecteurs[detecteur] << " ===" << std::endl;
		std::cout << "Flou = " << lire("Flou") / 10.0
			<< "  Ouverture = " << 2 * lire("Ouverture") + 3
			<< "  Seuil Sobel = " << lire("Sobel")
			<< "  Seuil Laplace = " << lire("Laplace")
			<< "  Canny = " << lire("Canny bas") << "/" << lire("Canny haut")
			<< "  Voisinage = " << voisinage << std::endl;
		std::cout << std::setw(12) << "Image" << std::setw(8) << "Detect" << std::setw(8) << "Ref"
			<< std::setw(8) << "Corr" << std::setw(8) << "FP" << std::setw(8) << "FN"
			<< std::setw(8) << "P" << std::setw(8) << "TFP" << std::setw(8) << "TFN" << std::endl;
	}

	for (int i = 0; i < images.size(); i++) {

		cv::Mat contours = detecterContours(images[i], detecteur);

		int detectes, contoursReference, corrects, fauxPositifs, fauxNegatifs;
		cv::Mat comparaison;
		evaluer(contours, references[i], voisinage,
			detectes, contoursReference, corrects, fauxPositifs, fauxNegatifs, comparaison);

		double total = corrects + fauxPositifs + fauxNegatifs;
		double P = corrects / total;
		double TFP = fauxPositifs / total;
		double TFN = fauxNegatifs / total;

		sommeP += P;
		sommeTFP += TFP;
		sommeTFN += TFN;

		if (details) {
			std::cout << std::setw(12) << noms[i] << std::setw(8) << detectes << std::setw(8) << contoursReference
				<< std::setw(8) << corrects << std::setw(8) << fauxPositifs << std::setw(8) << fauxNegatifs
				<< std::setw(8) << P << std::setw(8) << TFP << std::setw(8) << TFN << std::endl;

			cv::imwrite("resultats/" + noms[i] + "_" + nomsDetecteurs[detecteur] + ".png", 255 - contours);
			cv::imwrite("resultats/" + noms[i] + "_" + nomsDetecteurs[detecteur] + "_comparaison.png", comparaison);
		}
	}

	int n = images.size();
	std::cout << std::setw(12) << nomsDetecteurs[detecteur] << "  moyenne sur " << n << " images :"
		<< "  P = " << sommeP / n << "  TFP = " << sommeTFP / n << "  TFN = " << sommeTFN / n << std::endl;
}

int main() {

	std::vector<cv::Mat> images;
	std::vector<cv::Mat> references;

	for (int i = 0; i < noms.size(); i++) {
		cv::Mat image = cv::imread("images/" + noms[i] + ".pgm", 0);
		cv::Mat reference = cv::imread("images/gt/" + noms[i] + "_gt_binary.pgm", 0);
		if (image.empty() || reference.empty()) {
			std::cout << "Image introuvable : " << noms[i] << std::endl;
			return 0;
		}
		images.push_back(image);
		references.push_back(reference == 0);
	}

	std::cout << "Touches : l = traiter le lot avec le detecteur choisi" << std::endl;
	std::cout << "          t = comparer les 3 detecteurs sur le lot" << std::endl;
	std::cout << "          q = quitter" << std::endl;

	cv::namedWindow("Reglages");
	ajouterTrackbar("Image", 0, noms.size() - 1);
	ajouterTrackbar("Detecteur", 2, 2);
	ajouterTrackbar("Voisinage", 1, 1);
	ajouterTrackbar("Flou", 15, 50);
	ajouterTrackbar("Ouverture", 0, 2);
	ajouterTrackbar("Sobel", 70, 255);
	ajouterTrackbar("Laplace", 8, 127);
	ajouterTrackbar("Canny bas", 50, 300);
	ajouterTrackbar("Canny haut", 150, 600);

	while (true) {

		int numero = lire("Image");
		int detecteur = lire("Detecteur");

		cv::Mat contours = detecterContours(images[numero], detecteur);

		int detectes, contoursReference, corrects, fauxPositifs, fauxNegatifs;
		cv::Mat comparaison;
		evaluer(contours, references[numero], lire("Voisinage") == 1,
			detectes, contoursReference, corrects, fauxPositifs, fauxNegatifs, comparaison);

		double total = corrects + fauxPositifs + fauxNegatifs;

		cv::Mat info(300, 500, CV_8U, cv::Scalar(255));
		cv::putText(info, noms[numero] + " - " + nomsDetecteurs[detecteur], cv::Point(10, 30),
			cv::FONT_HERSHEY_PLAIN, 1.5, 0, 2);
		cv::putText(info, "P   = " + std::to_string(cvRound(100 * corrects / total)) + " %", cv::Point(10, 70),
			cv::FONT_HERSHEY_PLAIN, 1.5, 0, 2);
		cv::putText(info, "TFP = " + std::to_string(cvRound(100 * fauxPositifs / total)) + " %", cv::Point(10, 110),
			cv::FONT_HERSHEY_PLAIN, 1.5, 0, 2);
		cv::putText(info, "TFN = " + std::to_string(cvRound(100 * fauxNegatifs / total)) + " %", cv::Point(10, 150),
			cv::FONT_HERSHEY_PLAIN, 1.5, 0, 2);

		std::vector<std::string> legende = {
			"Detecteur : 0 Sobel, 1 Laplace, 2 Canny",
			"Voisinage : 0 pixel exact, 1 = 3x3",
			"Flou : sigma x 10 (15 = 1.5)",
			"Ouverture : 0 -> 3, 1 -> 5, 2 -> 7",
			"Sobel, Laplace : seuils (0-255, 0-127)"
		};
		cv::line(info, cv::Point(10, 170), cv::Point(410, 170), 128, 1);
		for (int i = 0; i < legende.size(); i++)
			cv::putText(info, legende[i], cv::Point(10, 195 + i * 24), cv::FONT_HERSHEY_PLAIN, 1.1, 80, 1);

		cv::imshow("Reglages", info);

		cv::namedWindow("Image");
		cv::imshow("Image", images[numero]);

		cv::namedWindow("Contours detectes");
		cv::imshow("Contours detectes", 255 - contours);

		cv::namedWindow("Reference");
		cv::imshow("Reference", 255 - references[numero]);

		cv::namedWindow("Comparaison");
		cv::imshow("Comparaison", comparaison);

		int touche = cv::waitKey(100);

		if (touche == 'q' || touche == 27)
			break;
		if (touche == 'l')
			traiterLot(images, references, detecteur, true);
		if (touche == 't') {
			std::cout << std::endl;
			for (int d = 0; d < 3; d++)
				traiterLot(images, references, d, false);
		}
	}

	return 0;
}
