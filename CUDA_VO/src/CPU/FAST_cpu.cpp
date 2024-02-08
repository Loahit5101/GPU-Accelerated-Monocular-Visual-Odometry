void show_image(cv::Mat img) {
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", img);
	cv::waitKey(0);
}


/**
 * @brief Generate 16 incremental indexes of pixels in surrounding circle.
 * 
 * @param circle output array
 * @param w width of data block (i.e. image width or shared mem width)
 */
void create_circle(int *circle, int w) {
	circle[0] = -3 * w;
	circle[1] = -3 * w + 1;
	circle[2] = -2 * w + 2;
	circle[3] = -w + 3;

	circle[4] = 3;
	circle[5] = w + 3;
	circle[6] = 2 * w + 2;
	circle[7] = 3 * w + 1;

	circle[8] = 3 * w;
	circle[9] = 3 * w - 1;
	circle[10] = 2 * w - 2;
	circle[11] = w - 3;

	circle[12] = -3;
	circle[13] = -w - 3;
	circle[14] = -2 * w - 2;
	circle[15] = -3 * w - 1;
}

/**
 * @brief Generate incremental indexes of mask used in non-maximal suppression
 * 
 * @param mask output array
 * @param w width of data block (i.e. image width or shared mem width)
 */
void create_mask(int *mask, int w) {
	// create mask with given defined mask size and width
	int start = -(int)MASK_SIZE / 2;
	int end = (int)MASK_SIZE / 2;
	int index = 0;
	for (int i = start; i <= end; i++)
	{
		for (int j = start; j <= end; j++)
		{
			mask[index] = i * w + j;
			index++;
		}
	}
}

/**
 * @brief Naive CPU implementation of FAST algorithm
 * 
 * @param input image in 1D array
 * @param scores helper array caching scores
 * @param mask 
 * @param circle 
 * @param width width of input
 * @param height height of input
 * @return std::vector<corner> vector of found corners
 */
std::vector<corner> cpu_FAST(unsigned char *input, unsigned *scores, int *mask, int *circle, int width, int height) {
	/// fast test
	std::vector<corner> ret;
	int id1d;
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			scores[id1d] = fast_test(input, circle, threshold, id1d);
		}
	}
	/// complex test
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			if (scores[id1d] > 0) {
				scores[id1d] = complex_test(input, scores, scores, circle, threshold, pi, id1d, id1d);
			}
		}
	}
	/// non-max suppression
	bool is_max;
	int val;
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			val = scores[id1d];
			if (val > 0) {
				is_max = true;
				for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
				{
					if (val < scores[id1d + mask[i]]) {
						is_max = false;
						break;
					}
				}
				if (is_max) {
					corner c;
					c.score = (unsigned)val;
					c.x = (unsigned)x;
					c.y = (unsigned)y;
					ret.push_back(c);
				}
			}
		}
	}
	return ret;
}

/**
 * @brief Parsing of main arguments
 * 
 * @param argc 
 * @param argv 
 */
void parse_args(int argc, char **argv){
	for (size_t i = 1; i < argc; i++)
	{
		std::string arg = std::string(argv[i]);
		if (arg == "-f") filename = argv[i + 1];
		if (arg == "-m") mode = atoi(argv[i + 1]);
		if (arg == "-p") pi = atoi(argv[i + 1]);
		if (arg == "-i") foto = true;
		if (arg == "-v") video = true;
		if (arg == "-t") threshold = atoi(argv[i + 1]);
		if (arg == "-c") circle_size = atoi(argv[i + 1]);
	}
	if (filename == NULL) {
		printf("\n--- Path to image must be specified in arguments ... quiting ---");
		exit(1);
	}
	if (mode < 0 || mode > 20) {
		printf("\n--- Mode must be in range 0 - 2 ... quiting ---");
		exit(1);
	}
	if (pi < 9 || pi > 12) {
		printf("\n--- Pi must be in range 9 - 12 ... quiting ---");
		exit(1);
	}
	if (threshold < 0 || threshold > 255) {
		printf("\n--- Threshold must be in range 0 - 255 ... quiting ---");
		exit(1);
	}
	printf("\n--- Runing with following setup: --- \n");
	printf("     Threshold: %d\n", threshold);
	printf("     Pi: %d\n", pi);
	printf("     Mode: %d\n", mode);
	printf("     File name: %s\n", filename);
	return;
}




/**
 * @brief Draw circles for all corners (with different color based on their score)
 * 
 * @param image 
 * @param corners found corners
 * @param number_of_corners 
 */
void write_circles(cv::Mat image, corner* corners, int number_of_corners) {
	/// draw corners 
	float start = (float)corners[number_of_corners - 1].score;
	float end = (float)corners[0].score;
	float rgb_k = 255 / (end - start);
	for (int i = 0; i < number_of_corners; i++)
	{
		unsigned inc = (corners[i].score - start)*rgb_k;
		cv::Scalar color = cv::Scalar(0, inc, 255 - inc);
		cv::circle(image, cv::Point(corners[i].x, corners[i].y), circle_size, color, 2);
	}
}



/**
 * @brief Method encapsulating FAST algorithm running on CPU
 * 
 * @param image 
 */
void run_on_cpu(cv::Mat image) {
	if (mode == 1) {
		std::vector<cv::KeyPoint> keypointsD;

		cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, true);
		detector->detect(image, keypointsD, cv::Mat());
		// cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < keypointsD.size(); i++) {
			cv::circle(image, keypointsD[i].pt, circle_size, cv::Scalar(0, 255, 0), 2);
		}
	}
	else {
		cv::Mat gray_img;	// create gray copy
		cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
		h_circle = (int*)malloc(CIRCLE_SIZE * sizeof(int));
		h_mask = (int*)malloc(MASK_SIZE*MASK_SIZE * sizeof(int));
		unsigned *h_scores = (unsigned*)malloc(image.cols*image.rows * sizeof(int));
		create_circle(h_circle, image.cols);
		create_mask(h_mask, image.cols);
		std::vector<corner> points = cpu_FAST(gray_img.data, h_scores, h_mask, h_circle, image.cols, image.rows);

		for (int i = 0; i < points.size(); i++) {
			cv::circle(image, cv::Point(points[i].x, points[i].y), circle_size, cv::Scalar(0, 255, 0), 2);
		}
	}

	//cv::Size size(1280, 720);	// resize for testing
	//resize(image, image, size);
	//show_image(image);
}

int main(int argc, char **argv)
{

	parse_args(argc, argv);


	run_on_cpu(image);
	cv::imwrite("output.jpg", image);
	printf("--- output.jpg generated ---");
		
	
	time_measured = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("--- output.avi generated in %f seconds ---", time_measured);

	
    return 0;
}
