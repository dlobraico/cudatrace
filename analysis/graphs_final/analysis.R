library(ggplot2) 
speed <- read.csv("output.csv", header=T)
speed$total_pixels = speed$xres * speed$yres
speed$real_seconds = speed$real_time/1000
speed$render_seconds = speed$render_time/1000
speed$megapixels = speed$total_pixels/1000000
speed$total_threads = speed$xthreads * speed$ythreads
levels(speed$total_threads) =  c(1,4,16,64,144,256,400,484) 

generate_speedup_factor <- function(df) {
    df$real_speedup = 1
    df$render_speedup = 1
    for (run in 1:3) {
        for (resolution in c(64,160,240,480,800,960,1120,1280,1440,2400,4800,8000,9600,11200,12800,14400,16000)) {
            cuda       <- df[df$run == run & (df$xres == resolution & df$yres == resolution) & df$application_type == "cuda",]
            sequential <- df[df$run == run & (df$xres == resolution & df$yres == resolution) & df$application_type == "sequential",]

            cuda$real_speedup   <- sequential[1,]$real_time/cuda$real_time
            cuda$render_speedup <- sequential[1,]$render_time/cuda$render_time

            df[df$run == run & (df$xres == resolution & df$yres == resolution) & df$application_type == "cuda",] <- cuda
            df[df$run == run & (df$xres == resolution & df$yres == resolution) & df$application_type == "sequential",] <- sequential
        }
    }
    return(df)
}

total_speed <- generate_speedup_factor(speed)
speed <- total_speed[speed$run == 1,]

p <- ggplot(speed, aes(megapixels, legend = TRUE)) +  
opts(title = "Performance of Raytracer Implementations at Various Resolutions and Block Sizes", legend.title = theme_text(colour='black', size=10, face='bold', hjust=-.1)) + 
geom_point(aes(y = real_time, shape = application_type, color = factor(total_threads)), size = 4) + scale_shape("Application Type", solid=TRUE) + scale_color_hue("Total Threads") + 
scale_y_continuous("Real Time (ms)", formatter="comma") +
scale_x_continuous("Megapixels Rendered", formatter="comma") 

p
ggsave("real_time.pdf")

p + 
scale_y_continuous("Real Time (ms)", formatter="comma", limits = c(0, 200000)) +
geom_line(aes(y = real_time, group = factor(total_threads), color = factor(total_threads)))
ggsave("real_time_zoom1.pdf")

p + 
scale_y_continuous("Real Time (ms)", formatter="comma", limits = c(0, 60000)) +
geom_line(aes(y = real_time, group = factor(total_threads), color = factor(total_threads)))
ggsave("real_time_zoom2.pdf")

p +
scale_y_continuous("Real Time (ms)", formatter="comma", limits = c(0, 18500)) + scale_x_continuous("Megapixels Rendered", formatter="comma", limits = c(0,25)) + 
geom_line(aes(y = real_time, group = factor(total_threads), color = factor(total_threads)))
ggsave("real_time_zoom3.pdf")

p <- ggplot(speed, aes(megapixels, legend = TRUE)) +  
opts(title = "Performance of Raytracer Implementations at Various Resolutions and Block Sizes", legend.title = theme_text(colour='black', size=10, face='bold', hjust=-.1)) + 
geom_point(aes(y = render_time, shape = application_type, color = factor(total_threads), size = factor(total_threads)), position=position_jitter(h=0,w=8)) + scale_shape("Application Type", solid=TRUE) + scale_color_hue("Total Threads") + scale_size_manual("Total Threads", values=c(2,3,4,8,10,12,14,16,5)) + 
scale_y_continuous("Render Time (ms)", formatter="comma") +
scale_x_continuous("Megapixels Rendered with Horizontal Jitter", formatter="comma") 
ggsave("render_time.pdf")

p + 
scale_y_continuous("Real Time (ms)", formatter="comma", limits = c(0, 100000))
ggsave("render_time_zoom.pdf")

p <- ggplot(speed, aes(megapixels, legend = TRUE)) +  
opts(title = "Performance of Raytracer Implementations at Various Resolutions and Block Sizes", legend.title = theme_text(colour='black', size=10, face='bold', hjust=-.1)) + 
geom_point(aes(y = render_time, shape = application_type, color = factor(total_threads)), size = 4) + 
scale_shape("Application Type", solid=TRUE) + 
scale_color_hue("Total Threads") + 
scale_x_continuous("Megapixels Rendered", formatter="comma", limits = c(0,65)) + 
scale_y_continuous("Render Time (ms)", formatter="comma", limits = c(0, 50000)) +
geom_smooth(aes(y = render_time, group = factor(total_threads), color = factor(total_threads), method="lm", fullrange=TRUE))
p
ggsave("render_time_no_jitter_with_line.pdf")

p <- ggplot(speed, aes(megapixels, legend = TRUE)) +  
opts(title = "Performance of Raytracer Implementations at Various Resolutions and Block Sizes", legend.title = theme_text(colour='black', size=10, face='bold', hjust=-.1)) + 
geom_point(aes(y = render_time, shape = application_type, color = factor(total_threads)), size = 4) + 
scale_shape("Application Type", solid=TRUE) + 
scale_color_hue("Total Threads") + 
scale_x_continuous("Megapixels Rendered", formatter="comma", limits = c(0,1.0)) + 
scale_y_continuous("Render Time (ms)", formatter="comma", limits = c(0, 1000)) #+
#geom_smooth(aes(y = render_time, group = factor(total_threads), method="lm", fullrange=TRUE))
p
ggsave("render_time_no_jitter_zoom1.pdf")

p +
scale_x_continuous("Megapixels Rendered", formatter="comma", limits = c(0,0.25)) + 
scale_y_continuous("Render Time (ms)", formatter="comma", limits = c(0, 200)) + 
geom_line(aes(y = render_time, group = factor(total_threads), color = factor(total_threads)))
ggsave("render_time_no_jitter_zoom2.pdf")

p +
scale_x_continuous("Megapixels Rendered", formatter="comma", limits = c(0,0.06)) + 
scale_y_continuous("Render Time (ms)", formatter="comma", limits = c(0, 50)) + 
geom_line(aes(y = render_time, group = factor(total_threads), color = factor(total_threads)))
ggsave("render_time_no_jitter_zoom3.pdf")


p <- ggplot(speed, aes(real_time, legend = TRUE)) +
opts(title = "Render Time vs. Real Time", legend.title = theme_text(colour = 'black', size=10, face='bold', hjust=-.1)) +
geom_point(aes(y=render_time, color = factor(total_threads), shape = application_type), size = 7) +
scale_y_continuous("Render Time (ms)", formatter="comma") +
scale_x_continuous("Real Time (ms)", formatter="comma") +
scale_color_hue("Total Threads") + scale_shape("Application Type", solid=FALSE)
p
ggsave("render_time_vs_real_time.pdf")

p + scale_y_continuous("Render Time (ms)", formatter="comma", limits = c(0,50000)) + scale_x_continuous("Real Time (ms)", formatter="comma", limits = c(0,50000))
ggsave("render_time_vs_real_time_zoom1.pdf")

p + scale_y_continuous("Render Time (ms)", formatter="comma", limits = c(0,1500)) + scale_x_continuous("Real Time (ms)", formatter="comma", limits = c(0,6000))
ggsave("render_time_vs_real_time_zoom2.pdf")

p <- ggplot(speed, aes(megapixels, legend = TRUE)) + 
geom_point(aes(y = speed$render_speedup, shape = application_type, color = factor(total_threads)), size = 4) +
scale_shape("Application Type", solid=TRUE) + 
scale_color_hue("Total Threads") + 
scale_y_continuous("Render Speedup") + 
scale_x_continuous("Megapixels Rendered", formatter="comma") + 
geom_line(aes(y = render_speedup, group = factor(total_threads), color = factor(total_threads))) 
p
ggsave("render_speedup_vs_megapixels.pdf")

p + scale_x_continuous("Megapixels Rendered", formatter="comma", limits = c(0,2))
ggsave("render_speedup_vs_megapixels_zoom.pdf")

p <- ggplot(speed, aes(megapixels, legend = TRUE)) + 
geom_point(aes(y = speed$real_speedup, shape = application_type, color = factor(total_threads)), size = 4) +
scale_shape("Application Type", solid=TRUE) + 
scale_color_hue("Total Threads") + 
scale_y_continuous("Real Speedup") + 
scale_x_continuous("Megapixels Rendered", formatter="comma") 
p + geom_line(aes(y = real_speedup, group = factor(total_threads), color = factor(total_threads))) 
p
ggsave("real_speedup_vs_megapixels.pdf")

p + scale_x_continuous("Megapixels Rendered", formatter="comma", limits = c(0,25)) +
scale_y_continuous("Real Speedup", limits = c(0, 2)) +
geom_smooth(aes(y = real_speedup, group = factor(total_threads), color = factor(total_threads))) 
ggsave("real_speedup_vs_megapixels_zoom.pdf")
