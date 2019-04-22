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

enum state {actif, inactif};



/********** micro BLAS LEVEL-1 + quelques fonctions non-standard **************/
static inline void copy(const double *x, double *y) // copy x dans y. x et y etant des vecteurs de dim3
{
	for (int i = 0; i < 3; i++)
		y[i] = x[i];
} 

static inline void zero(double *x) //x = 0. x vecteur 3D
{
	for (int i = 0; i < 3; i++)
		x[i] = 0;
} 

static inline void axpy(double alpha, const double *x, double *y) // y = y+ ax
{
	for (int i = 0; i < 3; i++)
		y[i] += alpha * x[i];
} 

static inline void scal(double alpha, double *x) // x = ax
{
	for (int i = 0; i < 3; i++)
		x[i] *= alpha;
} 

static inline double dot(const double *a, const double *b) //multiplication de vecteurs
{ 
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
} 

static inline double nrm2(const double *a) // norme du vecteur
{
	return sqrt(dot(a, a));
}

/********* fonction non-standard *************/
static inline void mul(const double *x, const double *y, double *z)// multiplication element par element
{
	for (int i = 0; i < 3; i++)
		z[i] = x[i] * y[i];
} 

static inline void normalize(double *x) // normalisation de X
{
	scal(1 / nrm2(x), x);
}

/* produit vectoriel */
static inline void cross(const double *a, const double *b, double *c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

/****** tronque *************/
static inline void clamp(double *x) // encadrement des valeurs des elements de X entre 0 et 1.
{
	for (int i = 0; i < 3; i++) {
		if (x[i] < 0)
			x[i] = 0;
		if (x[i] > 1)
			x[i] = 1;
	}
} 

/******************************* calcul des intersections rayon / sphere *************************************/
   
// returns distance, 0 if nohit 
double sphere_intersect(const struct Sphere *s, const double *ray_origin, const double *ray_direction) // renvoie la distance entre l'origine de rayon et le point d'intersection avec la sphere
																									   //nous permettant de connaitre le point d'intersection (point ray_origine + distance + ray_direction)
{ 
	double op[3];
	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
	copy(s->position, op);    // op devient la position de la sphere
	axpy(-1, ray_origin, op); //op est modifié, opn devient la distance entre s et ray origine
	double eps = 1e-4;
	double b = dot(op, ray_direction);
	double discriminant = b * b - dot(op, op) + s->radius * s->radius; 
	if (discriminant < 0)
		return 0;   /* pas d'intersection */
	else 
		discriminant = sqrt(discriminant);
	/* détermine la plus petite solution positive (i.e. point d'intersection le plus proche, mais devant nous) */
	double t = b - discriminant;
	if (t > eps) {
		return t;
	} else {
		t = b + discriminant;
		if (t > eps)
			return t;
		else
			return 0;  /* cas bizarre, racine double, etc. */
	}
}

/* détermine si le rayon intersecte l'une des spere; si oui renvoie true et fixe t, id */
bool intersect(const double *ray_origin, const double *ray_direction, double *t, int *id) //entre autre cherche la premiere boule 
{ 
	int n = sizeof(spheres) / sizeof(struct Sphere);
	double inf = 1e20; 
	*t = inf;
	for (int i = 0; i < n; i++) {
		double d = sphere_intersect(&spheres[i], ray_origin, ray_direction);
		if ((d > 0) && (d < *t)) {
			*t = d;
			*id = i;
		} 
	}
	return *t < inf;
} 

/* calcule (dans out) la lumiance reçue par la camera sur le rayon donné */
void radiance(const double *ray_origin, const double *ray_direction, int depth, unsigned short *PRNG_state, double *out)
{ 
	int id = 0;                             // id de la sphère intersectée par le rayon
	double t;                               // distance à l'intersection
	if (!intersect(ray_origin, ray_direction, &t, &id)) {
		zero(out);    // if miss, return black 
		return; 
	}
	const struct Sphere *obj = &spheres[id];
	
	/* point d'intersection du rayon et de la sphère */
	double x[3];
	copy(ray_origin, x);
	axpy(t, ray_direction, x);
	
	/* vecteur normal à la sphere, au point d'intersection */
	double n[3];  
	copy(x, n);
	axpy(-1, obj->position, n);
	normalize(n);
	
	/* vecteur normal, orienté dans le sens opposé au rayon 
	   (vers l'extérieur si le rayon entre, vers l'intérieur s'il sort) */
	double nl[3];
	copy(n, nl);
	if (dot(n, ray_direction) > 0)
		scal(-1, nl);
	
	/* couleur de la sphere */
	double f[3];
	copy(obj->color, f);
	double p = obj->max_reflexivity;

	/* processus aléatoire : au-delà d'une certaine profondeur,
	   décide aléatoirement d'arrêter la récusion. Plus l'objet est
	   clair, plus le processus a de chance de continuer. */
	depth++;
	if (depth > KILL_DEPTH) {
		if (erand48(PRNG_state) < p) {
			scal(1 / p, f); 
		} else {
			copy(obj->emission, out);
			return;
		}
	}

	/* Cas de la réflection DIFFuse (= non-brillante). 
	   On récupère la luminance en provenance de l'ensemble de l'univers. 
	   Pour cela : (processus de monte-carlo) on choisit une direction
	   aléatoire dans un certain cone, et on récupère la luminance en 
	   provenance de cette direction. */
	if (obj->refl == DIFF) {
		double r1 = 2 * M_PI * erand48(PRNG_state);  /* angle aléatoire */
		double r2 = erand48(PRNG_state);             /* distance au centre aléatoire */
		double r2s = sqrt(r2); 
		
		double w[3];   /* vecteur normal */
		copy(nl, w);
		
		double u[3];   /* u est orthogonal à w */
		double uw[3] = {0, 0, 0};
		if (fabs(w[0]) > .1)
			uw[1] = 1;
		else
			uw[0] = 1;
		cross(uw, w, u);
		normalize(u);
		
		double v[3];   /* v est orthogonal à u et w */
		cross(w, u, v);
		
		double d[3];   /* d est le vecteur incident aléatoire, selon la bonne distribution */
		zero(d);
		axpy(cos(r1) * r2s, u, d);
		axpy(sin(r1) * r2s, v, d);
		axpy(sqrt(1 - r2), w, d);
		normalize(d);
		
		/* calcule récursivement la luminance du rayon incident */
		double rec[3];
		radiance(x, d, depth, PRNG_state, rec);
		
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* dans les deux autres cas (réflection parfaite / refraction), on considère le rayon
	   réfléchi par la spère */

	double reflected_dir[3];
	copy(ray_direction, reflected_dir);
	axpy(-2 * dot(n, ray_direction), n, reflected_dir);

	/* cas de la reflection SPEculaire parfaire (==mirroir) */
	if (obj->refl == SPEC) { 
		double rec[3];
		/* calcule récursivement la luminance du rayon réflechi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* cas des surfaces diélectriques (==verre). Combinaison de réflection et de réfraction. */
	bool into = dot(n, nl) > 0;      /* vient-il de l'extérieur ? */
	double nc = 1;                   /* indice de réfraction de l'air */
	double nt = 1.5;                 /* indice de réfraction du verre */
	double nnt = into ? (nc / nt) : (nt / nc);
	double ddn = dot(ray_direction, nl);
	
	/* si le rayon essaye de sortir de l'objet en verre avec un angle incident trop faible,
	   il rebondit entièrement */
	double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
	if (cos2t < 0) {
		double rec[3];
		/* calcule seulement le rayon réfléchi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}
	
	/* calcule la direction du rayon réfracté */
	double tdir[3];
	zero(tdir);
	axpy(nnt, ray_direction, tdir);
	axpy(-(into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)), n, tdir);

	/* calcul de la réflectance (==fraction de la lumière réfléchie) */
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);
	double c = 1 - (into ? -ddn : dot(tdir, n));
	double Re = R0 + (1 - R0) * c * c * c * c * c;   /* réflectance */
	double Tr = 1 - Re;                              /* transmittance */
	
	/* au-dela d'une certaine profondeur, on choisit aléatoirement si
	   on calcule le rayon réfléchi ou bien le rayon réfracté. En dessous du
	   seuil, on calcule les deux. */
	double rec[3];
	if (depth > SPLIT_DEPTH) {
		double P = .25 + .5 * Re;             /* probabilité de réflection */
		if (erand48(PRNG_state) < P) {
			radiance(x, reflected_dir, depth, PRNG_state, rec);
			double RP = Re / P;
			scal(RP, rec);
		} else {
			radiance(x, tdir, depth, PRNG_state, rec);
			double TP = Tr / (1 - P); 
			scal(TP, rec);
		}
	} else {
		double rec_re[3], rec_tr[3];
		radiance(x, reflected_dir, depth, PRNG_state, rec_re);
		radiance(x, tdir, depth, PRNG_state, rec_tr);
		zero(rec);
		axpy(Re, rec_re, rec);
		axpy(Tr, rec_tr, rec);
	}
	/* pondère, prend en compte la luminance */
	mul(f, rec, out);
	axpy(1, obj->emission, out);
	return;
}

double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

int toInt(double x)
{
	return pow(x, 1 / 2.2) * 255 + .5;   /* gamma correction = 2.2 */
} 



double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/*double * sub_pixel_calc(double[3] PRNG_state,int sub_i, int sub_j, double[3] camera_direction, int samples){
	double subpixel_radiance[3] = {0, 0, 0};
	// simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne 
	for (int s = 0; s < samples; s++) { 
		// tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer 
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
		
		// estime la lumiance qui arrive sur la caméra par ce rayon 
		double sample_radiance[3];
		radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
		// fait la moyenne sur tous les rayons 
		axpy(1. / samples, sample_radiance, subpixel_radiance);
	}
	clamp(subpixel_radiance);
	return subpixel_radiance
}*/

void version1_static(int argc, char **argv){

	MPI_Init(&argc,&argv);
	int rank,size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	printf("%d : MPI init DONE \n", rank);

	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 200;
	int samples = 200;
	int nb_line = h/size;
	printf("hello i am %d\n", rank);

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
	double * image ;
	if(rank == 0)
		image = malloc(3 * w * h * sizeof(double));
	else
		image = malloc(3 * w * nb_line * sizeof(double));
		

	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}

	double t0 = my_gettimeofday();

	for (int i = nb_line *rank; i < nb_line *(rank+1); i++) {
 		unsigned short PRNG_state[3] = {0, 0, i*i*i};
		for (unsigned short j = 0; j < w; j++) {
			//printf(" precessus %d, pixel : %d - %d   -----  ",rank,i,j);
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
			//printf("%f %f %f \n",pixel_radiance[0], pixel_radiance[1], pixel_radiance[2]);
			//copy(pixel_radiance, image + 3 * ((h - 1 - (i - rank*nb_line) * w + j))); // <-- retournement vertical
			if(rank != 0){
				copy(pixel_radiance, image + 3 * ((nb_line - 1 - (i - rank*nb_line)) * w + j)); // <-- retournement vertical
			}
			else{
				copy(pixel_radiance, image + 3 * ((h - 1 - i) * w + j)); // <-- retournement vertical
			}
		}
	}

	double t1 = my_gettimeofday();
	printf("--------------------------------------\n");
	printf("     Processeur %d JOB FINISHED       \n",rank);
	printf("                time : %f             \n",t1-t0);
	printf("--------------------------------------\n");



	if (rank == 0){
		for(int k=size-1;k>0;k--){
			printf("%d : recieving image from %d \n",rank,k);
	       	MPI_Recv(&image[ nb_line* (size-1-k)*w*3],3*w*nb_line,MPI_DOUBLE,k,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	       }
	       //   stocke l'image dans un fichier au format NetPbm 
		{
			struct passwd *pass; 
			char nom_sortie[100] = "";
			char nom_rep[100] = "";

			pass = getpwuid(getuid()); 
			sprintf(nom_rep, "/nfs/home/sasl/eleves/main/3520621/Documents/HPC/Path_tracing_HPC/%s", pass->pw_name);
			mkdir(nom_rep, S_IRWXU);
			sprintf(nom_sortie, "%s/image.ppm", nom_rep);
			
			FILE *f = fopen(nom_sortie, "w");
			fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
			for (int i = 0; i < w * h; i++) 
		  		fprintf(f,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2])); 
			fclose(f); 
		}
	}
    else {
    	printf("%d : sending image \n",rank);
    	MPI_Send(image,3*w*nb_line,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    	printf("%d : image send \n",rank);
    }
	
	 

	free(image);

	MPI_Finalize();
}


bool verif(int* memory, int h){
	for(int i=0 ;  i<h ; i++){
		if (memory[i] == 0)
			return true;
	}
	return false;
}







void version2_dynamic(int argc, char **argv){
	MPI_Init(&argc,&argv);
	int rank,size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Request req[size];
	MPI_Request req_tab[size-1];
	MPI_Request send_req[size];
	int send_flag[size];	
	int flag_tab[size-1];
	MPI_Status status_tab;
	int flag[size];
	MPI_Status status;
	printf("%d : MPI init DONE \n", rank);

	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 200;
	int samples = 200;
	int line_number = 0;
	int nb_line = h/size;
	int line;
	int iter=0;
	if(rank ==0){
		printf("initial data : w = %d \nh = %d \nsamples = %d \nnb_line = %d ",w,h,samples,nb_line);
	}



	printf("hello i am %d\n", rank);

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
	double * image ;
	//double * final_image ;
	//int * image_map;
	if(rank == 0){
		
		image = malloc(3 * w * h * sizeof(double));
		
		//image_map = (int*)calloc(h,sizeof(int));
	}
	else
		image = malloc(3 * w * sizeof(double));
	//image = malloc(3 * w * sizeof(double));
		

	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}
	int * shared_memory;
	shared_memory = (int*)calloc(h,sizeof(int));

	int * shared_memory_tmp;
	shared_memory_tmp = (int*)calloc(h,sizeof(int));

	double * tab;
	tab = (double*)malloc((3*w + 1)*sizeof(double));

	int i = rank*nb_line;
	printf("proc %d initiate at line %d\n",rank,i);
	// for(int k =0 ; k<h ; k++){
	// 	if(k%nb_line == 0)
	// 		shared_memory[k] = 1;
	// }

	// for(int k=0 ; k<size ; k++){
	// 	shared_memory[k*nb_line] = 1;
	// }


	bool continuer = true;
	int count_line = 0;
	double t0 = my_gettimeofday();

	//for (int i = nb_line *rank; i < nb_line *(rank+1); i++) {
	while(continuer){

		// MPI_Irecv(shared_memory_tmp,h,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&req);

		// for(int k=0 ; k<h ; k++){
		// 	shared_memory[k] += shared_memory_tmp[k];
		// }
		// printf("proc %d recieve1  :", rank);
		// 	printf(" [ ");
		// 	for(int l=0 ; l<h ; l++ ){
		// 		printf("%d ",shared_memory[l] );
		// 	}
		// 	printf("] \n");
		
		continuer = verif(shared_memory, h);

		unsigned short PRNG_state[3] = {0, 0, i*i*i};
		for (unsigned short j = 0; j < w; j++) {
			//printf(" precessus %d, pixel : %d - %d   -----  ",rank,i,j);
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
			//printf("%f %f %f \n",pixel_radiance[0], pixel_radiance[1], pixel_radiance[2]);
			if(rank!=0){
				copy(pixel_radiance, image + 3 * j); // <-- retournement vertical
			}
			else{
				copy(pixel_radiance, image + 3*w*i+ 3 * j); // <-- retournement vertical
			}
		}




	
		tab[0] = (double)i;
		//printf("proc %d just done line %f or %d\n",rank, tab[0],i);

		

		if(line_number >= h){
			line_number = -1;
		}

		if (rank == 0){

			// for(int k=1 ; k<3*w+1 ; k++){
			// 	tab[k] = image[i*3*w + k-1];
			// }
			// line = tab[0];

	  //      	for(int k=1; k< 3*w+1; k++){
	  //      		image[(line*3*w + k -1) ] = tab[k]; 
	  //      	}
	       	// printf(" line done : %d \n",line);
	       	count_line++;
	       	printf("done by %d nb line done : %d, line %d \n",rank,count_line, i);


	       	for(int k=0 ; k<size-1 ; k++){

	       		if(iter > 0 && flag_tab[k] == 0){
					MPI_Request_free(&req_tab[k]);
				}

	       		MPI_Irecv(tab,3*w+1,MPI_DOUBLE,k+1,1,MPI_COMM_WORLD,&req_tab[k]);
		    
				MPI_Test(&req_tab[k],&flag_tab[k],&status_tab);
				if(flag_tab[k]){
					
					line = tab[0];
					//printf("%d recieve tab from %d with line %d \n",rank,status_tab.MPI_SOURCE, line);

			       	for(int l=1; l< 3*w+1; l++){
			       		image[(line*3*w + l -1) ] = tab[l]; 
			       	}
			       	count_line++;
			       	printf("done by %d nb line done : %d, line %d \n",status_tab.MPI_SOURCE,count_line, line);
				}
			}
	       	

	  //      	printf("proc %d tab  :", rank);
			// printf(" [ ");
			// for(int l=0 ; l<3*w ; l++ ){
			// 	printf("%f ",image[line*3*w + l ] );
			// }
			// printf("] \n");

	       	//printf("%d : recieving image line %d \n",rank,line);
		}
		else{
			// printf("proc %d image   :", rank);
			// printf(" [ ");
			// for(int l=0 ; l<3*w ; l++ ){
			// 	printf("%f ",image[l ] );
			// }
			// printf("] \n");


			for(int k=1 ; k<3*w+1 ; k++){
				tab[k] = image[k-1];
			}
			// MPI_Request tmp_reg;
			// MPI_Isend(tab,3*w+1,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&tmp_reg);
			MPI_Send(tab,3*w+1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
			
		}

		//MPI_Cancel(&req);
		for(int k=0 ; k < size-1 ; k++){
			// if(req[k] != MPI_REQUEST_NULL ){
			// 	MPI_Request_free(&req[k]);
			// }
			//MPI_Cancel(&req[k]);
			if(iter > 0 && flag[k] == 0){
				//MPI_Cancel(&req[k]);
				MPI_Request_free(&req[k]);
			}
			MPI_Irecv(shared_memory_tmp,h,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&req[k]);
			//MPI_Recv(shared_memory_tmp,h,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&req);

			MPI_Test(&req[k],&flag[k],&status);

			if(flag[k]){
				printf("%d recieve shared memory from %d \n",rank,status.MPI_SOURCE);
				for(int l=0 ; l<h ; l++){
					shared_memory[l] += shared_memory_tmp[l];
					if(shared_memory[l] > 0){
						shared_memory[l] = 1;
					}
				}
			}
		}

		
		 
		// printf("proc %d recieve1  :", rank);
		// printf(" [ ");
		// for(int l=0 ; l<h ; l++ ){
		// 	printf("%d ",shared_memory_tmp[l] );
		// }
		// printf("] \n");

		for(int l=0 ; l<h ; l++ ){
			if(shared_memory[(l + rank*nb_line)%h] == 0){	
				i = (l + rank*nb_line)%h;
				shared_memory[i] = 1;
				break;
			}
			// else if(l == h-1){
			// 	i = -1;
			// }
			if(l == h){
				continuer = false;
			}
		}




		printf("proc %d shared memory  :", rank);
		printf(" [ ");
		for(int l=0 ; l<h ; l++ ){
			printf("%d ",shared_memory[l] );
		}
		printf("] \n");

		
		for(int k=0 ; k<size ; k++){
			if(k != rank)
				// if(iter>0){
				// 	MPI_Status send_status;
				// 	MPI_Test(&send_req[k],&send_flag[k],&send_status);
				// 	if(send_flag[k] == 0){
				// 		MPI_Cancel(&send_req[k]);
				// 	}
				// }
				//MPI_Isend(shared_memory,h,MPI_INT,k,0,MPI_COMM_WORLD,&send_req[k]);
				//MPI_Send(shared_memory,h,MPI_INT,k,0,MPI_COMM_WORLD);
				MPI_Bsend(shared_memory,h,MPI_INT,k,0,MPI_COMM_WORLD);
		}
		//MPI_Irecv(shared_memory_tmp,h,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&req);



		// printf("proc %d bcasting (send):", rank);
		// printf(" [ ");
		// for(int l=0 ; l<h ; l++ ){
		// 	printf("%d ",shared_memory[l] );
		// }
		// printf("] \n ");


		continuer = verif(shared_memory, h);
		
		iter++;

		
	
	}

	double t1 =my_gettimeofday();
	printf("--------------------------------------\n");
	printf("     Processeur %d JOB FINISHED       \n",rank);
	printf("                time : %f             \n",t1-t0);
	printf("--------------------------------------\n");





	if (rank == 0){
		// MPI_Request final_req;
		// int final_flag;
		// while(count_line <= h){
		// 	MPI_Irecv(tab,3*w+1,MPI_DOUBLE,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&final_req);
		    
		// 	MPI_Test(&final_req,&final_flag,&status_tab);
		// 	if(final_flag){
		// 		//printf("%d recieve tab from %d \n",rank,status_tab.MPI_SOURCE);
		// 		line = tab[0];

		//        	for(int k=1; k< 3*w+1; k++){
		//        		image[(line*3*w + k -1) ] = tab[k]; 
		//        	}
		//        	count_line++;
		//        	printf("nb line done : %d \n",count_line);
		//        	//image_map[line] = 1;
		// 	}
		// }


		for(int k=0 ; k<size ; k++){
			if(k != rank)
				MPI_Send(shared_memory,h,MPI_INT,k,0,MPI_COMM_WORLD);
		}
		printf( "proc 0 saving image \n");
		double * reverse_image ;
		reverse_image = malloc(3 * w * h * sizeof(double));
		for (int k=h-1 ; k>=0 ; k--){
			for (int l=0 ; l<w*3 ; l++){
				reverse_image[(h-(k+1))*3*w +l] = image[k*3*w + l];
			}
		}

		
		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[100] = "";

		pass = getpwuid(getuid()); 
		sprintf(nom_rep, "/nfs/home/sasl/eleves/main/3520621/Documents/HPC/Path_tracing_HPC/%s", pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/beta_image%d.ppm", nom_rep,size);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
		for (int i = 0; i < w * h; i++) 
	  		fprintf(f,"%d %d %d ", toInt(reverse_image[3 * i]), toInt(reverse_image[3 * i + 1]), toInt(reverse_image[3 * i + 2])); 
		fclose(f);
		free(reverse_image); 
		printf( "image saved as %s \n", nom_sortie);
		//free(final_image);
		//free(image_map);
	}


	free(image);
	free(tab);
	free(shared_memory);

	MPI_Finalize();
}

void version2_beta_dynamic(int argc, char **argv){
	MPI_Init(&argc,&argv);
	int rank,size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Request req;
	MPI_Request req_tab[size-1];
	MPI_Request send_req[size];
	int send_flag[size];	
	int flag_tab[size-1];
	MPI_Status status_tab;
	int flag;
	MPI_Status status;
	printf("%d : MPI init DONE \n", rank);

	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 200;
	int samples = 200;
	int line_number = 0;
	int nb_line = h/size;
	int line;
	int iter=0;
	int state;
	if(rank ==0){
		printf("initial data : w = %d \nh = %d \nsamples = %d \nnb_line = %d ",w,h,samples,nb_line);
	}



	printf("hello i am %d\n", rank);

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
	double * image ;
	//double * final_image ;
	//int * image_map;
	if(rank == 0){
		
		image = malloc(3 * w * h * sizeof(double));
		
		//image_map = (int*)calloc(h,sizeof(int));
	}
	else
		image = malloc(3 * w * sizeof(double));
	//image = malloc(3 * w * sizeof(double));
		

	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}
	int * shared_memory;
	shared_memory = (int*)calloc(h,sizeof(int));

	int * shared_memory_tmp;
	shared_memory_tmp = (int*)calloc(h,sizeof(int));

	double * tab;
	tab = (double*)malloc((3*w + 1)*sizeof(double));




	bool continuer = true;
	int count_line = 0;
	double t0 = my_gettimeofday();
	int i;

	if (rank == 0){
		i = 0;
		shared_memory[i] = 1;
		state = actif;
	}





	while(continuer){
		if(iter == 0){
			if(rank > 0){
				MPI_Recv(shared_memory_tmp,h,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				for(int l=0 ; l<h ; l++){
					shared_memory[l] += shared_memory_tmp[l];
					if(shared_memory[l] > 0){
						shared_memory[l] = 1;
					}
				}
				for(int k=0 ; k<h ; k++ ){
					if(shared_memory[k] == 0){
						i = k;
						shared_memory[i] = 1;
						state = actif;
						break;
					}
					else if(k == h - 1){
						continuer = false;
						state = inactif;
					}
				}
				if(rank != size -1){
					//MPI_Bsend(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					//printf("proc %d : shared memory send to %d\n", rank, rank +1);
				}
				else{
					//MPI_Bsend(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
					MPI_Send(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
					//printf("proc %d : shared memory send to %d\n", rank, 0);
				}
				printf("proc %d : first recieve\n",rank );
				state = actif;
			}
			else{
				MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
			}

		}


		if(iter > 0){
			MPI_Test(&req,&flag,&status);

			if(flag){
				printf("%d I_recieve shared memory from %d \n",rank,status.MPI_SOURCE);
				for(int l=0 ; l<h ; l++){
					shared_memory[l] += shared_memory_tmp[l];
					if(shared_memory[l] > 0){
						shared_memory[l] = 1;
					}
				}
				for(int k=0 ; k<h ; k++ ){
					if(shared_memory[k] == 0){
						i = k;
						shared_memory[i] = 1;
						state = actif;
						break;
					}
					else if(k == h - 1){
						continuer = false;
						state = inactif;
					}
				}
				if(rank != size -1){
					//MPI_Bsend(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					printf("proc %d : shared memory send to %d\n", rank, rank +1);
				}
				else{
					//MPI_Bsend(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
					MPI_Send(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
					printf("proc %d : shared memory send to %d\n", rank, 0);
				}

			}
			else{
				MPI_Request_free(&req);
				if(i + size < h){
					i += size;
					shared_memory[i] = 1;
					state = actif;
				}
				else{
					// if(state = actif)
					// 	state = inactif;
					//MPI_wait(&req, &status);
					if(rank != size -1){
						//MPI_Bsend(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
						MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
						printf("proc %d :  last send to %d\n", rank, rank +1);
					}
					else{
						//MPI_Bsend(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
						MPI_Send(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
						printf("proc %d : last send to %d\n", rank, 0);
					}
					if(!verif(shared_memory,h) ){
						MPI_Recv(shared_memory_tmp,h,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						printf("%d Recieve shared memory from %d \n",rank,status.MPI_SOURCE);
						for(int l=0 ; l<h ; l++){
							shared_memory[l] += shared_memory_tmp[l];
							if(shared_memory[l] > 0){
								shared_memory[l] = 1;
							}
						}
						for(int k=0 ; k<h ; k++ ){
							if(shared_memory[k] == 0){
								i = k;
								shared_memory[i] = 1;
								state = actif;
								break;
							}
							else if(k == h - 1){
								continuer = false;
								state = inactif;
							}
						}
					}
					else{
						continuer = false;
						state =inactif;
					}
					if(state == actif){

						if(rank != size -1){
							//MPI_Bsend(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
							MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
							printf("proc %d : shared memory send to %d\n", rank, rank +1);
						}
						else{
							//MPI_Bsend(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
							MPI_Send(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
							printf("proc %d : shared memory send to %d\n", rank, 0);
						}
					}

				}
			}
		}
		// if(iter > 0 && flag == 0){
		// 	MPI_Request_free(&req);
		// }
		
		if(rank > 0){
			MPI_Irecv(shared_memory_tmp,h,MPI_INT,rank-1,0,MPI_COMM_WORLD,&req);
		}
		else{
			MPI_Irecv(shared_memory_tmp,h,MPI_INT,size-1,0,MPI_COMM_WORLD,&req);
		}


		// MPI_Test(&req,&flag,&status);

		// if(flag){
		// 	printf("%d recieve shared memory from %d \n",rank,status.MPI_SOURCE);
		// 	for(int l=0 ; l<h ; l++){
		// 		shared_memory[l] += shared_memory_tmp[l];
		// 		// if(shared_memory[l] > 0){
		// 		// 	shared_memory[l] = 1;
		// 		// }
		// 	}
		// 	for(int k=0 ; k<h ; k++ ){
		// 		if(shared_memory[k] == 0){
		// 			i = k;
		// 			shared_memory[i] = 1;
		// 			state = actif;
		// 			break;
		// 		}
		// 		else if(k == h - 1){
		// 			continuer = false;
		// 			state = inactif;
		// 		}
		// 	}
		// }
		// else{
		// 	if(i + size < h){
		// 		i += size;
		// 		shared_memory[i] = 1;
		// 		state = actif;
		// 	}
		// 	else{
		// 		if(state = actif)
		// 			state = inactif;
		// 		//MPI_wait(&req, &status);

		// 	}
		// }

		
		
		printf("proc %d shared memory  :", rank);
		printf(" [ ");
		for(int l=0 ; l<h ; l++ ){
			printf("%d ",shared_memory[l] );
		}
		printf("] \n");



		if(state == actif){
			unsigned short PRNG_state[3] = {0, 0, i*i*i};
			for (unsigned short j = 0; j < w; j++) {
				//printf(" precessus %d, pixel : %d - %d   -----  ",rank,i,j);
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
				//printf("%f %f %f \n",pixel_radiance[0], pixel_radiance[1], pixel_radiance[2]);
				if(rank!=0){
					copy(pixel_radiance, image + 3 * j); // <-- retournement vertical
				}
				else{
					copy(pixel_radiance, image + 3*w*i+ 3 * j); // <-- retournement vertical
					
				}
			}
			if(rank==0){
				count_line++;
				printf("done by %d nb line done : %d, line %d \n",rank,count_line, i);
			}
		

			


			
			if(rank != 0){
				tab[0] = (double)i;

				for(int k=1 ; k<3*w+1 ; k++){
					tab[k] = image[k-1];
				}
				// MPI_Request tmp_reg;
				// MPI_Isend(tab,3*w+1,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&tmp_reg);
				MPI_Send(tab,3*w+1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
				printf("proc %d just did line %d \n", rank,i);
				
			}
		}
		if (rank == 0){

	       	for(int k=0 ; k<size-1 ; k++){
	       		

	       		if(iter > 0){
	       			MPI_Test(&req_tab[k],&flag_tab[k],&status_tab);
	       			if(flag_tab[k]){
						line = tab[0];
						//printf("%d recieve tab from %d with line %d \n",rank,status_tab.MPI_SOURCE, line);

				       	for(int l=1; l< 3*w+1; l++){
				       		image[(line*3*w + l -1) ] = tab[l]; 
				       	}
				       	count_line++;
				       	printf("done by %d nb line done : %d, line %d \n",status_tab.MPI_SOURCE,count_line, line);
					}
				
					else{
						
				       	MPI_Request_free(&req_tab[k]);
					}
				}

	       		MPI_Irecv(tab,3*w+1,MPI_DOUBLE,k+1,1,MPI_COMM_WORLD,&req_tab[k]);
		    
				
				// if(flag_tab[k]){
					
					
				// }
			}
		}
		iter++;
		state = inactif;
	}

	double t1 =my_gettimeofday();
	printf("--------------------------------------\n");
	printf("     Processeur %d JOB FINISHED       \n",rank);
	printf("                time : %f             \n",t1-t0);
	printf("--------------------------------------\n");





	if (rank == 0){
		// MPI_Request final_req;
		// int final_flag;
		// while(count_line <= h){
		// 	MPI_Irecv(tab,3*w+1,MPI_DOUBLE,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&final_req);
		    
		// 	MPI_Test(&final_req,&final_flag,&status_tab);
		// 	if(final_flag){
		// 		//printf("%d recieve tab from %d \n",rank,status_tab.MPI_SOURCE);
		// 		line = tab[0];

		//        	for(int k=1; k< 3*w+1; k++){
		//        		image[(line*3*w + k -1) ] = tab[k]; 
		//        	}
		//        	count_line++;
		//        	printf("nb line done : %d \n",count_line);
		//        	//image_map[line] = 1;
		// 	}
		// }


		for(int k=0 ; k<size ; k++){
			if(k != rank)
				MPI_Send(shared_memory,h,MPI_INT,k,0,MPI_COMM_WORLD);
		}
		printf( "proc 0 saving image \n");
		double * reverse_image ;
		reverse_image = malloc(3 * w * h * sizeof(double));
		for (int k=h-1 ; k>=0 ; k--){
			for (int l=0 ; l<w*3 ; l++){
				reverse_image[(h-(k+1))*3*w +l] = image[k*3*w + l];
			}
		}

		
		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[100] = "";

		pass = getpwuid(getuid()); 
		sprintf(nom_rep, "/nfs/home/sasl/eleves/main/3520621/Documents/HPC/Path_tracing_HPC/%s", pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/beta_image%d.ppm", nom_rep,size);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
		for (int i = 0; i < w * h; i++) 
	  		fprintf(f,"%d %d %d ", toInt(reverse_image[3 * i]), toInt(reverse_image[3 * i + 1]), toInt(reverse_image[3 * i + 2])); 
		fclose(f);
		free(reverse_image); 
		printf( "image saved as %s \n", nom_sortie);
		//free(final_image);
		//free(image_map);
	}


	free(image);
	free(tab);
	free(shared_memory);

	MPI_Finalize();
}


void version2_beta_dynamic_simple(int argc, char **argv){
	MPI_Init(&argc,&argv);
	int rank,size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Request req;
	MPI_Status status;
	printf("%d : MPI init DONE \n", rank);

	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 200;
	int samples = 200;



	int line_number = 0;
	int nb_line = h/size;
	int line;
	int iter=0;
	int state;
	int end_count =0;
	if(rank ==0){
		printf("initial data : w = %d \nh = %d \nsamples = %d \nnb_line = %d ",w,h,samples,nb_line);
	}



	printf("hello i am %d\n", rank);

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
	double * image ;
	double * image_tmp;
	if(rank == 0){
		image_tmp = (double*)calloc(3*w*h,sizeof(double));
	}
		
	image = (double*)calloc(3*w*h,sizeof(double));
				

	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}
	int * shared_memory;
	shared_memory = (int*)calloc(h,sizeof(int));

	int * shared_memory_tmp;
	shared_memory_tmp = (int*)calloc(h,sizeof(int));


	bool continuer = true;
	int count_line = 0;
	double t0 = my_gettimeofday();
	int i;

	if (rank == 0){
		i = 0;
		shared_memory[i] = 1;
		state = actif;
	}

	while(continuer){
		/* premier tour de boucle, initialisation des envois*/
		if(iter == 0){
			if(rank > 0){
				MPI_Recv(shared_memory_tmp,h,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				for(int l=0 ; l<h ; l++){
					shared_memory[l] += shared_memory_tmp[l];
					if(shared_memory[l] > 0){
						shared_memory[l] = 1;
					}
				}
				for(int k=0 ; k<h ; k++ ){
					if(shared_memory[k] == 0){
						i = k;
						shared_memory[i] = 1;
						state = actif;
						break;
					}
					else if(k == h - 1){
						continuer = false;
						state = inactif;
					}
				}
				if(rank != size -1){
					MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
				}
				else{
					MPI_Send(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
				}
				printf("proc %d : first recieve\n",rank );
				state = actif;
			}
			else{
				MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
			}

		}
		else{ /*Recoit -> choisit -> envoie ->calcul*/
			MPI_Recv(shared_memory_tmp,h,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			for(int l=0 ; l<h ; l++){
				shared_memory[l] += shared_memory_tmp[l];
				if(shared_memory[l] > 0){
					shared_memory[l] = 1;
				}
			}
			for(int k=0 ; k<h ; k++ ){
				if(shared_memory[k] == 0){
					i = k;
					shared_memory[i] = 1;
					state = actif;
					break;
				}
				else if(k == h - 1){
					end_count++;
					state = inactif;
					continuer = false;
					if(rank != size -1){
						MPI_Bsend(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
					}
					else{
						MPI_Bsend(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
					}
				}
			}
			if(end_count == 0){
				if(rank != size -1){
					MPI_Send(shared_memory,h,MPI_INT,rank+1,0,MPI_COMM_WORLD);
				}
				else{
					MPI_Send(shared_memory,h,MPI_INT,0,0,MPI_COMM_WORLD);
				}
			}

		}
		


	
		
		
		printf("proc %d shared memory  :", rank);
		printf(" [ ");
		for(int l=0 ; l<h ; l++ ){
			printf("%d ",shared_memory[l] );
		}
		printf("] \n");



		if(state == actif){
			unsigned short PRNG_state[3] = {0, 0, i*i*i};
			for (unsigned short j = 0; j < w; j++) {
				//printf(" precessus %d, pixel : %d - %d   -----  ",rank,i,j);
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
				copy(pixel_radiance, image + 3*w*i+ 3 * j); // <-- retournement vertical
			}
		}

		iter++;
		state = inactif;
	}

	double t1 =my_gettimeofday();
	printf("--------------------------------------\n");
	printf("     Processeur %d JOB FINISHED       \n",rank);
	printf("                time : %f             \n",t1-t0);
	printf("--------------------------------------\n");




	if (rank == 0){
		for(int k=1 ; k<size ; k++){
			MPI_Recv(image_tmp,3*w*h,MPI_DOUBLE,k,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			for(int i=0 ; i<3*w*h ; i++){
				image[i] += image_tmp[i];
			}
		}
	


		printf( "proc 0 saving image \n");
		double * reverse_image ;
		reverse_image = malloc(3 * w * h * sizeof(double));
		for (int k=h-1 ; k>=0 ; k--){
			for (int l=0 ; l<w*3 ; l++){
				reverse_image[(h-(k+1))*3*w +l] = image[k*3*w + l];
			}
		}

		
		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[100] = "";

		pass = getpwuid(getuid()); 
		sprintf(nom_rep, "/nfs/home/sasl/eleves/main/3520621/Documents/HPC/Path_tracing_HPC/%s", pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/beta_image%d.ppm", nom_rep,size);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
		for (int i = 0; i < w * h; i++) 
	  		fprintf(f,"%d %d %d ", toInt(reverse_image[3 * i]), toInt(reverse_image[3 * i + 1]), toInt(reverse_image[3 * i + 2])); 
		fclose(f);
		free(reverse_image); 
		printf( "image saved as %s \n", nom_sortie);
		free(image_tmp);
	}
	else{
		MPI_Send(image,3*w*h,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
		
	}
	free(image);
	//free(tab);
	free(shared_memory);

	MPI_Finalize();
}


void traitement_token(int rank, int size,int token, bool work, int *state, bool *continuer, int *i, int work_limit){
	int *token_send;
	if(token  == -2){
		*token_send = token;
		if(rank < size -1)
			MPI_Send(token_send,1,MPI_INT,rank+1,2,MPI_COMM_WORLD);
		else
			MPI_Send(token_send,1,MPI_INT,0,2,MPI_COMM_WORLD);
	}
	else if(token >= 0){
		if(work){
			int line_left = work_limit - *i;
			*i++;
			*token_send = *i;
			MPI_Send(token_send,1,MPI_INT,token,0,MPI_COMM_WORLD);
			*i++;
		}
		else{
			if(token == rank){
				// recieving his own message -> exit : token = -1
				*token_send = -1;
				if(rank < size -1)
					MPI_Send(token_send,1,MPI_INT,rank+1,2,MPI_COMM_WORLD);
				else
					MPI_Send(token_send,1,MPI_INT,0,2,MPI_COMM_WORLD);
			}
			else{
				// no work to give -> pass the token
				*token_send = token;
				if(rank < size -1)
					MPI_Send(token_send,1,MPI_INT,rank+1,2,MPI_COMM_WORLD);
				else
					MPI_Send(token_send,1,MPI_INT,0,2,MPI_COMM_WORLD);
			}
		}
	}
	else if(token == -1){
		*token_send = token;
		if(rank < size -1)
			MPI_Send(token_send,1,MPI_INT,rank+1,2,MPI_COMM_WORLD);
		else
			MPI_Send(token_send,1,MPI_INT,0,2,MPI_COMM_WORLD);
		state = inactif;
		continuer = false;
	}
	
}



void version3_dynamic_ring_token(int argc, char **argv){
	MPI_Init(&argc,&argv);
	int rank,size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//MPI_Request req;
	MPI_Status status;
	int flag;
	printf("%d : MPI init DONE \n", rank);

	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 20;
	int samples = 200;
	bool work =true;
	bool continuer = true;



	int line_number = 0;
	int nb_line = h/size;
	int line;
	int iter=0;
	int state;
	int end_count =0;
	int token = -10; //-10 means no token
	int work_limit = (rank +1)*nb_line ;
	if(rank ==0){
		printf("initial data : w = %d \nh = %d \nsamples = %d \nnb_line = %d \n",w,h,samples,nb_line);
	}



	printf("hello i am %d\n", rank);

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
	double * image ;
	double * image_tmp;
	if(rank == 0){
		image_tmp = (double*)calloc(3*w*h,sizeof(double));
		token = -2;
	}
		
	image = (double*)calloc(3*w*h,sizeof(double));
				

	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}
	

	int i = rank * nb_line;
	int *token_send;


	double t0 = my_gettimeofday();





	while(continuer){
		MPI_Iprobe(MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&flag,&status);

		if(flag && !work){
			printf("proc %d recieving msg",rank);
			int * i_tmp;
			MPI_Recv(i_tmp,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			i = *i_tmp;
			work == true;
		}


		if(!work){
			printf("proc %d entering no work zone. \n",rank);
			if(token == -10){
				printf("proc %d no token, waiting for it\n",rank);
				// int * token_tmp;
				// if(rank > 0)
				// 	MPI_Recv(token_tmp,1,MPI_INT,rank-1,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				// else
				// 	MPI_Recv(token_tmp,1,MPI_INT,size-1,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

				// token = *token_tmp;
			}
			printf("proc %d : token is here\n",rank);

			if(token == -2){
				printf("proc %d  token -2 , sending",rank);
				token = rank;
				token_send = &token;
				if(rank < size -1)
					MPI_Send(token_send,1,MPI_INT,rank+1,2,MPI_COMM_WORLD);
				else
					MPI_Send(token_send,1,MPI_INT,0,2,MPI_COMM_WORLD);

				token = -10;
			}
			else{
				printf("proc %d  token here , checking",rank);
				traitement_token(rank, size, token, work, &state, &continuer, &i, work_limit);
				token = -10;
			}
		}




		if(rank > 0)
			MPI_Iprobe(rank -1,2,MPI_COMM_WORLD,&flag,&status);
		else
			MPI_Iprobe(size -1,2,MPI_COMM_WORLD,&flag,&status);

		if(flag){
			MPI_Recv(&token,1,MPI_INT,status.MPI_SOURCE,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			traitement_token(rank, size, token, work, &state, &continuer, &i, work_limit);
		}





		if(state == actif){
			unsigned short PRNG_state[3] = {0, 0, i*i*i};
			for (unsigned short j = 0; j < w; j++) {
				//printf(" precessus %d, pixel : %d - %d   -----  ",rank,i,j);
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
				copy(pixel_radiance, image + 3*w*i+ 3 * j); // <-- retournement vertical
			}
			printf( "proc %d did line %d \n",rank,i);
		}

		i++;
		if(i >= work_limit){
			work = false;
		}

		iter++;
		//state = inactif;
	}

	double t1 =my_gettimeofday();
	printf("--------------------------------------\n");
	printf("     Processeur %d JOB FINISHED       \n",rank);
	printf("                time : %f             \n",t1-t0);
	printf("--------------------------------------\n");




	if (rank == 0){
		for(int k=1 ; k<size ; k++){
			MPI_Recv(image_tmp,3*w*h,MPI_DOUBLE,k,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			for(int i=0 ; i<3*w*h ; i++){
				image[i] += image_tmp[i];
			}
		}
	


		printf( "proc 0 saving image \n");
		double * reverse_image ;
		reverse_image = malloc(3 * w * h * sizeof(double));
		for (int k=h-1 ; k>=0 ; k--){
			for (int l=0 ; l<w*3 ; l++){
				reverse_image[(h-(k+1))*3*w +l] = image[k*3*w + l];
			}
		}

		
		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[100] = "";

		pass = getpwuid(getuid()); 
		sprintf(nom_rep, "/nfs/home/sasl/eleves/main/3520621/Documents/HPC/Path_tracing_HPC/%s", pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/beta_image%d.ppm", nom_rep,size);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
		for (int i = 0; i < w * h; i++) 
	  		fprintf(f,"%d %d %d ", toInt(reverse_image[3 * i]), toInt(reverse_image[3 * i + 1]), toInt(reverse_image[3 * i + 2])); 
		fclose(f);
		free(reverse_image); 
		printf( "image saved as %s \n", nom_sortie);
		free(image_tmp);
	}
	else{
		MPI_Send(image,3*w*h,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
		
	}
	free(image);
	//free(tab);
	//free(shared_memory);

	MPI_Finalize();
}




int main(int argc, char **argv)
{
	printf("BEGIN\n");

	//version2_dynamic(argc, argv);
	//version1_static(argc, argv);
	//version2_beta_dynamic(argc, argv);
	//version2_beta_dynamic_simple(argc, argv);
	version3_dynamic_ring_token(argc, argv);

	return 0;

}