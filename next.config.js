/** @type {import('next').NextConfig} */
const nextConfig = {
  basePath: "/mushrooms",
  rewrites: async () => {
    const backendUrl =
      process.env.NEXT_PUBLIC_API_URL ||
      (process.env.NODE_ENV === "development"
        ? "http://127.0.0.1:9998"
        : "http://mushroom-classifier-backend:9998");

    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
      {
        source: "/docs",
        destination: `${backendUrl}/docs`,
      },
      {
        source: "/openapi.json",
        destination: `${backendUrl}/openapi.json`,
      },
    ];
  },
};

module.exports = nextConfig;
