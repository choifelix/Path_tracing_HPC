/* basé sur on smallpt, a Path Tracer by Kevin Beason, 2008
 *  	http://www.kevinbeason.com/smallpt/ 
 *	
 * Code séquentiel Converti en C et modifié par Charles Bouillaguet, 2019
 *
 * Parallélisation effectué par Antoine Daflon et Félix Choi, 2019
 *
 * Pour des détails sur le processus de rendu, lire :
 * 	https://docs.google.com/open?id=0B8g97JkuSSBwUENiWTJXeGtTOHFmSm51UC01YWtCZw
 */



#include "pathtracer.h"
#include <mpi.h>

double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

double * sub_pixel_calc(double[3] PRNG_state,int sub_i, int sub_j, double[3] camera_direction, int samples){
	double subpixel_radiance[3] = {0, 0, 0};
	/* simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne */
	for (int s = 0; s < samples; s++) { 
		/* tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer */
		double r1 = 2 * erand48(PRNG_state);
		double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
		double r2 = 2 * erand48(PRNG_state);
		double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
		double ray_direction[3];
		copy(camera_direction, ray_direction);
		axpy(((sub_i + .5 + dy) / 2 + i) / h - .5, cy, ray_direction);
		axpy(((sub_j + .5 + dx) / 2 + j) / w - .5, cx, ray_direction);
		normalize(ray_direction);

		double ray_origin[3];
		copy(camera_position, ray_origin);
		axpy(140, ray_direction, ray_origin);
		
		/* estime la lumiance qui arrive sur la caméra par ce rayon */
		double sample_radiance[3];
		radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
		/* fait la moyenne sur tous les rayons */
		axpy(1. / samples, sample_radiance, subpixel_radiance);
	}
	clamp(subpixel_radiance);
	return subpixel_radiance
}

void version1_nul(){

	MPI_Init(&argc,&argv);
	int rank,size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 200;
	int samples = 200;
	nb_line = h/size;

	/* Gros cas test (big, slow and pretty): */
	/* int w = 3840; */
	/* int h = 2160; */
	/* int samples = 5000;  */

	if (argc == 2) 
		samples = atoi(argv[1]) / 4;

	static const double CST = 0.5135;  /* ceci défini l'angle de vue */
	double camera_position[3] = {50, 52, 295.6};
	double camera_direction[3] = {0, -0.042612, -1};
	normalize(camera_direction);

	/* incréments pour passer d'un pixel à l'autre */
	double cx[3] = {w * CST / h, 0, 0};    
	double cy[3];
	cross(cx, camera_direction, cy);  /* cy est orthogonal à cx ET à la direction dans laquelle regarde la caméra */
	normalize(cy);
	scal(CST, cy);

	/* précalcule la norme infinie des couleurs */
	int n = sizeof(spheres) / sizeof(struct Sphere);
	for (int i = 0; i < n; i++) {
		double *f = spheres[i].color;
		if ((f[0] > f[1]) && (f[0] > f[2]))
			spheres[i].max_reflexivity = f[0]; 
		else {
			if (f[1] > f[2])
				spheres[i].max_reflexivity = f[1];
			else
				spheres[i].max_reflexivity = f[2]; 
		}
	}

	/* boucle principale */
	if(rank>0){
		double *image = malloc(3 * w * nb_line * sizeof(*image));
	}
	else{
		double *image = malloc(3 * w * h * sizeof(*image));
	}

	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}

	for (int i = nb_line *rank; i < nb_line *(rang+1); i++) {
 		unsigned short PRNG_state[3] = {0, 0, i*i*i};
		for (unsigned short j = 0; j < w; j++) {
			/* calcule la luminance d'un pixel, avec sur-échantillonnage 2x2 */
			double pixel_radiance[3] = {0, 0, 0};
			for (int sub_i = 0; sub_i < 2; sub_i++) {
				for (int sub_j = 0; sub_j < 2; sub_j++) {
					double subpixel_radiance[3] = {0, 0, 0};
					/* simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne */
					for (int s = 0; s < samples; s++) { 
						/* tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer */
						double r1 = 2 * erand48(PRNG_state);
						double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
						double r2 = 2 * erand48(PRNG_state);
						double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						double ray_direction[3];
						copy(camera_direction, ray_direction);
						axpy(((sub_i + .5 + dy) / 2 + i) / h - .5, cy, ray_direction);
						axpy(((sub_j + .5 + dx) / 2 + j) / w - .5, cx, ray_direction);
						normalize(ray_direction);

						double ray_origin[3];
						copy(camera_position, ray_origin);
						axpy(140, ray_direction, ray_origin);
						
						/* estime la lumiance qui arrive sur la caméra par ce rayon */
						double sample_radiance[3];
						radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
						/* fait la moyenne sur tous les rayons */
						axpy(1. / samples, sample_radiance, subpixel_radiance);
					}
					clamp(subpixel_radiance);
					/* fait la moyenne sur les 4 sous-pixels */
					axpy(0.25, subpixel_radiance, pixel_radiance);
				}
			}
			copy(pixel_radiance, image + 3 * ((h - 1 - i) * w + j)); // <-- retournement vertical
		}
	}

	if (rank>0){
    	MPI_Send(image,3*w*nb_line,MPI_UNSIGNED_DOUBLE,0,0,MPI_COMM_WORLD);
    }
    else {
	    for(int k=1;k<size;k++){

	       MPI_Recv(image+ (nb_lin*w*3),3*w*nb_line,MPI_UNSIGNED_DOUBLE,k,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	    }
	}
	    /* stocke l'image dans un fichier au format NetPbm */
	{
		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[30] = "";

		pass = getpwuid(getuid()); 
		sprintf(nom_rep, "/tmp/%s", pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/image.ppm", nom_rep);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
		for (int i = 0; i < w * h; i++) 
	  		fprintf(f,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2])); 
		fclose(f); 
	}

	free(image);

}









int main(int argc, char **argv)
{


	MPI_Init(&argc,&argv);
	int rank,size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//assert((h%size)==0);

	MPI_Status status;

	if (rank>0 && rank < size/4){
	   	MPI_Send(ima,w*h/size,MPI_UNSIGNED_CHAR,0,0,MPI_COMM_WORLD);

	}

	else {
    	for(int k=1;k<size;k++){

   //    MPI_Recv(ima+((k*w*h)/size),w*h/size,MPI_UNSIGNED_CHAR,k,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	    }
	}
	
	/* debut du chronometrage */
  	debut = my_gettimeofday();

  	fin = my_gettimeofday();
 	fprintf( stderr, "Temps total de calcul : %g sec\n", fin - debut);
  	fprintf( stdout, "%g\n", fin - debut);
 

  	MPI_Finalize();
}