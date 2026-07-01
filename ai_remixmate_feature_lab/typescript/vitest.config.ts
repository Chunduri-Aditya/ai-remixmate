export default {
  test: {
    environment: "node",
    include: ["src/tests/**/*.test.ts"]
  },
  cacheDir: ".vitest-cache"
};
